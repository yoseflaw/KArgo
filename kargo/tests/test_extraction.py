import unittest
from kargo import logger, terms
from pke.unsupervised import PositionRank
terms.log.setLevel(logger.WARNING)


class TestExtraction(unittest.TestCase):

    def setUp(self) -> None:
        self.core_nlp_folder = "../../data/test/core_nlp_samples"

    def test_positionrank(self):
        positionrank_extractor = terms.PKEBasedTermsExtractor(PositionRank)
        # using default settings
        positionrank_keyphrases = positionrank_extractor.extract(
            self.core_nlp_folder, 10,
            grammar= r"""
                NBAR:
                    {<NOUN|PROPN|NUM|ADJ>*<NOUN|PROPN>}

                NP:
                    {<NBAR>}
                    {<NBAR><ADP><NBAR>}
                """,
            filtering_params={},
            weighting_params={}
        )
        self.assertEqual(3, len(positionrank_keyphrases))

    def test_embedrank(self):
        embedrank_extractor = terms.EmbedRankTermsExtractor(
            emdib_model_path="../../pretrain_models/torontobooks_unigrams.bin",
        )
        embedrank_keyphrases = embedrank_extractor.extract(
            self.core_nlp_folder, 10,
            grammar=r"""
                NALL:
                    {<NN|NNP|NNS|NNPS>}

                NBAR:
                    {<NALL|CD|JJ>*<NALL>}

                NP:
                    {<NBAR>}
                    {<NBAR><IN><NBAR>}
                """,
            considered_tags={'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'IN', 'CD'}
        )
        self.assertEqual(3, len(embedrank_keyphrases))


if __name__ == "__main__":
    unittest.main(warnings="ignore")
