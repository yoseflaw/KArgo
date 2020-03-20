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
        all_precisions = self.evaluator.calculate_precision_all()
        avg_precision_score = evaluation.Evaluator.get_average_scores(all_precisions)
        for scores in avg_precision_score.values():
            self.assertEqual(len(scores), sum(scores))

    def test_recall(self):
        all_relative_recalls = self.evaluator.calculate_relative_recalls_all()
        avg_relative_recalls = evaluation.Evaluator.get_average_scores(all_relative_recalls)
        for scores in avg_relative_recalls.values():
            self.assertEqual(len(scores), sum(scores))


if __name__ == "__main__":
    unittest.main()
