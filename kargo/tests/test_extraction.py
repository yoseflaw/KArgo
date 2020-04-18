import unittest
import warnings

from kargo.corpus import Corpus
from kargo import logger, extraction
from pke.unsupervised import PositionRank
extraction.log.setLevel(logger.WARNING)


class TestExtraction(unittest.TestCase):

    def setUp(self) -> None:
        self.core_nlp_folder = "../../data/test/core_nlp_samples"

    def test_positionrank(self):
        positionrank_extractor = extraction.PKEBasedExtractor(PositionRank)
        # using default settings
        positionrank_keyphrases = positionrank_extractor.extract(
            self.core_nlp_folder, 10, selection_params={}, weighting_params={}
        )
        self.assertEqual(2, len(positionrank_keyphrases))

    def test_embedrank(self):
        embedrank_extractor = extraction.EmbedRankExtractor(
            emdib_model_path="../../pretrain_models/torontobooks_unigrams.bin",
        )
        embedrank_keyphrases = embedrank_extractor.extract(self.core_nlp_folder, 10)
        self.assertEqual(2, len(embedrank_keyphrases))


if __name__ == "__main__":
    unittest.main(warnings="ignore")
