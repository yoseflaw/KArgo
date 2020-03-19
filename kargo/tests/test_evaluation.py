import unittest
from kargo import evaluation, logger
from kargo.corpus import Corpus
evaluation.log.setLevel(logger.WARNING)


class TestEvalution(unittest.TestCase):

    def setUp(self) -> None:
        labelled_corpus = Corpus("../../data/test/samples_with_terms.xml")
        self.evaluator = evaluation.Evaluator(labelled_corpus)
        self.accurate_preds = self.evaluator.true_terms

    def test_precision(self):
        all_predictions = self.evaluator.calculate_precision_all(self.accurate_preds)
        for _, score in all_predictions.items():
            self.assertEqual(score, 1.0)

    def test_recall(self):
        all_relative_recalls = self.evaluator.calculate_relative_recalls_all(
            [self.accurate_preds, self.accurate_preds]
        )
        for _, scores in all_relative_recalls.items():
            self.assertEqual(len(scores), sum(scores))


if __name__ == "__main__":
    unittest.main()
