import os
from tqdm import tqdm
from csv import DictWriter, DictReader
from corpus import Corpus
from pke.unsupervised import PositionRank
from pke.readers import MinimalCoreNLPReader
from swisscom_ai.research_keyphrase.embeddings.emb_distrib_local import EmbeddingDistributorLocal
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.method import MMRPhrase
from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingCoreNLP

from kargo import logger
log = logger.get_logger(__name__, logger.INFO)


class Extractor(object):

    @staticmethod
    def write_terms_to(all_terms, output_file):
        all_terms_reformat = []
        with open(output_file, "w") as csv_output:
            fieldnames = ["document_id", "terms"]
            csv_writer = DictWriter(csv_output, fieldnames)
            csv_writer.writeheader()
            for document_id in all_terms:
                csv_writer.writerow({"document_id": document_id, "terms": "|".join(all_terms[document_id])})
        return all_terms_reformat

    @staticmethod
    def read_terms_from(input_file):
        all_terms = {}
        with open(input_file, "r") as csv_input:
            reader = DictReader(csv_input)
            for row in reader:
                all_terms[row["document_id"]] = row["terms"].split("|")
        return all_terms


class PKEBasedExtractor(Extractor):

    def __init__(self, extractor_class, **extractor_init_params):
        self.extractor_class = extractor_class
        self.extractor_init_params = extractor_init_params

    def extract(self, core_nlp_folder, n_term, selection_params, weighting_params, output_file=None):
        xml_files = [filename for filename in os.listdir(core_nlp_folder) if filename.endswith(".xml")]
        all_terms = {}
        for xml_file in tqdm(xml_files):
            extractor = self.extractor_class(**self.extractor_init_params)
            extractor.load_document(input=os.path.join(core_nlp_folder, xml_file), language="en")
            extractor.candidate_selection(**selection_params)
            extractor.candidate_weighting(**weighting_params)
            terms = [term for term, _ in extractor.get_n_best(n_term)]
            document_id = xml_file.split(".")[0]
            all_terms[document_id] = terms
        if output_file: Extractor.write_terms_to(all_terms, output_file)
        return all_terms


class EmbedRankExtractor(Extractor):

    def __init__(self, emdib_model_path):
        self.embedding_distrib = EmbeddingDistributorLocal(emdib_model_path)

    def extract(self, core_nlp_folder, n_term, considered_tags=None, lang="en",
                beta=0.55, alias_threshold=0.7, output_file=None):
        xml_files = [filename for filename in os.listdir(core_nlp_folder) if filename.endswith(".xml")]
        all_terms = {}
        for xml_file in tqdm(xml_files):
            core_nlp_reader = MinimalCoreNLPReader()
            core_nlp_doc = core_nlp_reader.read(path=os.path.join(core_nlp_folder, xml_file))
            tagged_text = [list(zip(sentence.words, sentence.pos)) for sentence in core_nlp_doc.sentences]
            text_obj = InputTextObj(tagged_text, lang)
            if considered_tags: text_obj.considered_tags = considered_tags
            result = MMRPhrase(self.embedding_distrib, text_obj,
                               N=n_term, beta=beta, alias_threshold=alias_threshold)
            document_id = xml_file.split(".")[0]
            all_terms[document_id] = result[0]
        if output_file: Extractor.write_terms_to(all_terms, output_file)
        return all_terms


if __name__ == "__main__":
    n = 10
    core_nlp_folder = "../data/processed/stanford_core_nlp_xmls"
    positionrank_extractor = PKEBasedExtractor(PositionRank)
    positionrank_selection_params = {
        "grammar": r"""
                    NBAR:
                        {<NOUN|PROPN|NUM|ADJ>*<NOUN|PROPN>}

                    NP:
                        {<NBAR>}
                        {<NBAR><ADP><NBAR>}
                    """,
        "maximum_word_number": 5
    }
    positionrank_terms = positionrank_extractor.extract(
        core_nlp_folder, n,
        selection_params=positionrank_selection_params,
        weighting_params={},
        output_file="../results/extracted_terms/positionrank.csv"
    )
    embedrank_extractor = EmbedRankExtractor(
        emdib_model_path="../pretrain_models/torontobooks_unigrams.bin"
    )
    embedrank_terms = embedrank_extractor.extract(
        core_nlp_folder, n,
        considered_tags={"NOUN", "PROPN", "NUM", "ADJ", "ADP"},
        output_file="../results/extracted_terms/embedrank.csv"
    )

