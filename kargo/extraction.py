import os

from corpus import Corpus
from pke.unsupervised import PositionRank
from swisscom_ai.research_keyphrase.embeddings.emb_distrib_local import EmbeddingDistributorLocal
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.method import MMRPhrase
from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingCoreNLP

from kargo import logger
log = logger.get_logger(__name__, logger.INFO)


class PKEBasedExtractor(object):

    def __init__(self, extractor_class):
        self.extractor_class = extractor_class
        self.extractor = extractor_class()

    def extract(self, input_folder, n_keyphrase, selection_params):
        xml_files = [filename for filename in os.listdir(input_folder) if filename.endswith(".xml")]
        all_keyphrases = {}
        for xml_file in xml_files:
            self.extractor.load_document(input=os.path.join(input_folder, xml_file), language="en")
            self.extractor.candidate_selection(**selection_params)
            self.extractor.candidate_weighting()
            keyphrases = [keyphrase for keyphrase, _ in self.extractor.get_n_best(n_keyphrase)]
            document_id = xml_file.split(".")[0]
            all_keyphrases[document_id] = keyphrases
        return all_keyphrases


class EmbedRankExtractor(object):

    def __init__(self, emdib_model_path, core_nlp_host, core_nlp_port):
        self.embedding_distrib = EmbeddingDistributorLocal(emdib_model_path)
        self.pos_tagger = PosTaggingCoreNLP(core_nlp_host, core_nlp_port)

    def extract(self, corpus, n_keyphrase, lang="en", beta=0.55, alias_threshold=0.7):
        all_keyphrases = {}
        for document in corpus.iter_documents():
            document_id = document.document_id.text
            title = document.title.text
            content = " ".join([p.text for p in document.content.p])
            raw_text = title + "\n" + content
            tagged_text = self.pos_tagger.pos_tag_raw_text(raw_text)
            text_obj = InputTextObj(tagged_text, lang)
            result = MMRPhrase(self.embedding_distrib, text_obj,
                               N=n_keyphrase, beta=beta, alias_threshold=alias_threshold)
            all_keyphrases[document_id] = result[0]
        return all_keyphrases


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
    positionrank_keyphrases = positionrank_extractor.extract(core_nlp_folder, n, positionrank_selection_params)
    # print(position_rank_keyphrases)
    corpus = Corpus("../data/processed/random_sample_annotated.xml")
    embedrank_extractor = EmbedRankExtractor(
        emdib_model_path="../pretrain_models/torontobooks_unigrams.bin",
        core_nlp_host="localhost",
        core_nlp_port=9000
    )
    embedrank_keyphrases = embedrank_extractor.extract(corpus, n)
    combined_keypharses = {}
    for document in corpus.iter_documents():
        document_id = document.document_id.text
        combined_keypharses[document_id] = {
            "manual": [term.text for term in document.terms.term],
            "positionrank": positionrank_keyphrases[document_id],
            "embedrank": embedrank_keyphrases[document_id]
        }
    from pprint import pprint
    pprint(combined_keypharses)
