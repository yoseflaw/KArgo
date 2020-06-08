import os
from csv import DictWriter, DictReader

import numpy as np
import nltk
from pke import compute_document_frequency, load_document_frequency_file
from tqdm import tqdm
from pke.unsupervised import PositionRank, MultipartiteRank, KPMiner, TfIdf, YAKE
from pke.readers import MinimalCoreNLPReader
from spacy.lang.en.stop_words import STOP_WORDS

from swisscom_ai.research_keyphrase.embeddings.emb_distrib_local import EmbeddingDistributorLocal
from swisscom_ai.research_keyphrase.model.extractor import unique_ngram_candidates, GRAMMAR_EN
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.method import _MMR

from kargo import logger
log = logger.get_logger(__name__, logger.INFO)


class TermsExtractor(object):

    @staticmethod
    def write_terms_to(all_terms, output_file):
        with open(output_file, "w") as csv_output:
            fieldnames = ["document_id", "terms"]
            csv_writer = DictWriter(csv_output, fieldnames)
            csv_writer.writeheader()
            for document_id in all_terms:
                csv_writer.writerow({"document_id": document_id, "terms": "|".join(all_terms[document_id])})
        return True

    @staticmethod
    def read_terms_from(input_file):
        all_terms = {}
        with open(input_file, "r") as csv_input:
            reader = DictReader(csv_input)
            for row in reader:
                all_terms[row["document_id"]] = row["terms"].split("|")
        return all_terms


class PKEBasedTermsExtractor(TermsExtractor):

    def __init__(self, extractor_class, **extractor_init_params):
        self.extractor_class = extractor_class
        self.extractor_init_params = extractor_init_params

    def candidate_selection(self, extractor, grammar=None):
        extractor.grammar_selection(grammar=grammar)

    def candidate_filtering(self, extractor,
                            stoplist=None,
                            minimum_length=3,
                            minimum_word_size=0,
                            valid_punctuation_marks='-',
                            maximum_word_number=5,
                            only_alphanum=False,
                            pos_blacklist=None,
                            offset_cutoff=None,
                            min_frequency=0,
                            strip_outer_stopwords=False):

        """
        (From PKE lib)
        :param extractor: PKE extractor based on LoadFile class
        :param stoplist: list of strings, defaults to None.
        :param minimum_length: minimum number of characters for a
                candidate, defaults to 3.
        :param minimum_word_size: minimum number of characters for a
                token to be considered as a valid word, defaults to 2.
        :param valid_punctuation_marks: punctuation marks that are valid
                for a candidate, defaults to '-'.
        :param maximum_word_number: maximum length in words of the
                candidate, defaults to 5.
        :param only_alphanum: filter candidates containing non (latin)
                alpha-numeric characters, defaults to True.
        :param pos_blacklist: list of unwanted Part-Of-Speeches in
                candidates, defaults to None.
        :param offset_cutoff: the number of words after which candidates are
                filtered out, defaults to None.
        :param min_frequency: least allowable seen frequency, defaults to 0.
        :param strip_outer_stopwords: remove any stopwords at the last position.
        :return: None
        """
        extractor.candidate_filtering(
            stoplist,
            minimum_length,
            minimum_word_size,
            valid_punctuation_marks,
            maximum_word_number,
            only_alphanum,
            pos_blacklist
        )
        _stoplist = stoplist if stoplist else []
        for k in list(extractor.candidates):
            v = extractor.candidates[k]
            if offset_cutoff is not None and v.offsets[0] > offset_cutoff:
                del extractor.candidates[k]
            elif min_frequency is not None and len(v.surface_forms) < min_frequency:
                del extractor.candidates[k]
            # for YAKE
            elif strip_outer_stopwords \
                    and v.surface_forms[0][0].lower() in _stoplist \
                    or v.surface_forms[0][-1].lower() in _stoplist \
                    or len(v.surface_forms[0][0]) < 3 \
                    or len(v.surface_forms[0][-1]) < 3:
                del extractor.candidates[k]

    def extract(self, core_nlp_folder, n_term, grammar, filtering_params, weighting_params, output_file=None):
        xml_files = [filename for filename in os.listdir(core_nlp_folder) if filename.endswith(".xml")]
        all_terms = {}
        for xml_file in tqdm(xml_files):
            extractor = self.extractor_class(**self.extractor_init_params)
            extractor.load_document(input=os.path.join(core_nlp_folder, xml_file), language="en")
            self.candidate_selection(extractor, grammar=grammar)
            self.candidate_filtering(extractor, **filtering_params)
            extractor.candidate_weighting(**weighting_params)
            terms = [term for term, _ in extractor.get_n_best(n_term)]
            document_id = xml_file.split(".")[0]
            all_terms[document_id] = terms
        if output_file: TermsExtractor.write_terms_to(all_terms, output_file)
        return all_terms


class EmbedRankTermsExtractor(TermsExtractor):

    def __init__(self, emdib_model_path):
        self.embedding_distrib = EmbeddingDistributorLocal(emdib_model_path)

    def candidate_selection(self, grammar, embdistrib, text_obj):
        def extract_candidates(no_subset=False):
            keyphrase_candidate = set()
            np_parser = nltk.RegexpParser(grammar if grammar else GRAMMAR_EN)
            trees = np_parser.parse_sents(text_obj.pos_tagged)
            for tree in trees:
                for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                    keyphrase_candidate.add(' '.join(word for word, tag in subtree.leaves()))
            keyphrase_candidate = {kp for kp in keyphrase_candidate if len(kp.split()) <= 5}
            if no_subset:
                keyphrase_candidate = unique_ngram_candidates(keyphrase_candidate)
            else:
                keyphrase_candidate = list(keyphrase_candidate)
            return keyphrase_candidate

        def extract_candidates_embedding_for_doc():
            c = np.array(extract_candidates(text_obj))
            if len(c) > 0:
                embeddings = np.array(embdistrib.get_tokenized_sents_embeddings(c))
                valid_candidates_mask = ~np.all(embeddings == 0, axis=1)
                return c[valid_candidates_mask], embeddings[valid_candidates_mask, :]
            else:
                return np.array([]), np.array([])

        candidates, candidate_embs = extract_candidates_embedding_for_doc()
        return candidates, candidate_embs

    def extract(self, core_nlp_folder, n_term, grammar, considered_tags=None, lang="en",
                beta=0.55, alias_threshold=0.7, output_file=None):
        xml_files = [filename for filename in os.listdir(core_nlp_folder) if filename.endswith(".xml")]
        all_terms = {}
        for xml_file in tqdm(xml_files):
            core_nlp_reader = MinimalCoreNLPReader()
            core_nlp_doc = core_nlp_reader.read(path=os.path.join(core_nlp_folder, xml_file))
            tagged_text = [list(zip(sentence.words, sentence.pos)) for sentence in core_nlp_doc.sentences]
            text_obj = InputTextObj(tagged_text, lang)
            if considered_tags: text_obj.considered_tags = considered_tags
            candidates, candidate_embs = self.candidate_selection(grammar, self.embedding_distrib, text_obj)
            if len(candidates) > 0:
                result = _MMR(self.embedding_distrib, text_obj, candidates, candidate_embs,
                              N=n_term, beta=beta, use_filtered=True, alias_threshold=alias_threshold)
            else:
                result = (None, None, None)
            document_id = xml_file.split(".")[0]
            all_terms[document_id] = result[0]
        if output_file: TermsExtractor.write_terms_to(all_terms, output_file)
        return all_terms


def run_trial():
    n = 10
    snlp_folder = "../data/test/core_nlp_samples"
    compute_document_frequency(
        snlp_folder, os.path.join("../data/test/interim/test_cargo_df.tsv.gz"),
        stoplist=list(STOP_WORDS)
    )
    cargo_df = load_document_frequency_file("../data/test/interim/test_cargo_df.tsv.gz")
    pke_factory = {
        "grammar":  r"""
                NBAR:
                    {<NOUN|PROPN|NUM|ADJ>*<NOUN|PROPN>}

                NP:
                    {<NBAR>}
                    {<NBAR><ADP><NBAR>}
                """,
        "filtering_params": {
            "stoplist": list(STOP_WORDS)
        },
        "extractors": {
            "tfidf": {
                "instance": PKEBasedTermsExtractor(TfIdf),
                "weighting_params": {"df": cargo_df}
            },
            "yake": {
                "instance": PKEBasedTermsExtractor(YAKE),
                "filtering_params": {
                    "only_alphanum": True,
                    "strip_outer_stopwords": True
                },
                "weighting_params": {
                    "stoplist": list(STOP_WORDS)
                }
            },
            "kpm": {
                "instance": PKEBasedTermsExtractor(KPMiner),
                "weighting_params": {"df": cargo_df}
            },
            "mprank": {
                "instance": PKEBasedTermsExtractor(MultipartiteRank),
                "weighting_params": {}
            },
            "positionrank": {
                "instance": PKEBasedTermsExtractor(PositionRank),
                "weighting_params": {}
            }
        }
    }
    for name in pke_factory["extractors"]:
        log.info(f"Begin Extraction with PKE based extractor: {name}")
        extractor_instance = pke_factory["extractors"][name]["instance"]
        if "filtering_params" in pke_factory["extractors"][name]:
            filtering_params = {
                **pke_factory["filtering_params"],
                **pke_factory["extractors"][name]["filtering_params"]
            }
        else:
            filtering_params = pke_factory["filtering_params"]
        extractor_instance.extract(
            snlp_folder, n,
            grammar=pke_factory["grammar"],
            filtering_params=filtering_params,
            weighting_params=pke_factory["extractors"][name]["weighting_params"],
            output_file=os.path.join("../data/test/extracted_terms_sample/", f"{name}.csv")
        )
    log.info("Begin Extraction with EmbedRank")
    embedrank_extractor = EmbedRankTermsExtractor(
        emdib_model_path="../pretrain_models/torontobooks_unigrams.bin"
    )
    embedrank_terms = embedrank_extractor.extract(
        snlp_folder, n,
        grammar=r"""
                NALL:
                    {<NN|NNP|NNS|NNPS>}

                NBAR:
                    {<NALL|CD|JJ>*<NALL>}

                NP:
                    {<NBAR>}
                    {<NBAR><IN><NBAR>}
                """,
        considered_tags={'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'IN', 'CD'},
        output_file="../data/test/extracted_terms_sample/embedrank.csv"
    )


if __name__ == "__main__":
    run_trial()
