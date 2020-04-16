import unittest
from kargo import evaluation, logger
from kargo.corpus import Corpus
evaluation.log.setLevel(logger.WARNING)


class TestEvalution(unittest.TestCase):

    def setUp(self) -> None:
        labelled_corpus = Corpus("../../data/test/samples_with_terms.xml")
        self.evaluator = evaluation.Evaluator(labelled_corpus)
        accurate_preds = {
            document_id: self.evaluator.true_terms[document_id][:10] for document_id in self.evaluator.true_terms
        }
        self.evaluator.add_prediction("method1", accurate_preds)
        self.evaluator.add_prediction("method2", accurate_preds)

    def test_precision(self):
        precisions = self.evaluator.calculate_precision_all()
        precisions_df = evaluation.Evaluator.get_aggr_scores_df(precisions, "precisions")
        self.assertEqual(len(precisions_df), sum(precisions_df["precisions"]))

    def test_recall(self):
        relative_recalls = self.evaluator.calculate_relative_recalls_all()
        relative_recalls_df = evaluation.Evaluator.get_aggr_scores_df(relative_recalls, "relative recalls")
        self.assertEqual(len(relative_recalls_df), sum(relative_recalls_df["relative recalls"]))


if __name__ == "__main__":
    unittest.main()
