import unittest
import warnings

from kargo.corpus import Corpus
from kargo import logger, extraction
from pke.unsupervised import PositionRank
extraction.log.setLevel(logger.WARNING)


class TestExtraction(unittest.TestCase):

    def setUp(self) -> None:
        warnings.simplefilter("ignore", ResourceWarning)

    def test_positionrank(self):
        core_nlp_folder = "../../data/test/core_nlp_samples"
        positionrank_extractor = extraction.PKEBasedExtractor(PositionRank)
        # using default settings
        positionrank_selection_params = {}
        positionrank_keyphrases = positionrank_extractor.extract(core_nlp_folder, 10, positionrank_selection_params)
        self.assertEqual(len(positionrank_keyphrases), 3)

    def test_embedrank(self):
        embedrank_extractor = extraction.EmbedRankExtractor(
            emdib_model_path="../../pretrain_models/torontobooks_unigrams.bin",
            core_nlp_host="localhost",
            core_nlp_port=9000
        )
        embedrank_extractor.pos_tagger.parser.session.close()
        corpus = Corpus("../../data/test/samples_with_terms.xml")
        embedrank_keyphrases = embedrank_extractor.extract(corpus, 10)
        self.assertEqual(len(embedrank_keyphrases), 3)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
