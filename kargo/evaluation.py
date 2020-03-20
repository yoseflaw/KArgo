import pandas as pd
import altair as alt
from kargo import logger
from kargo.corpus import Corpus
log = logger.get_logger(__name__, logger.INFO)


class Evaluator(object):

    def __init__(self, reference_corpus: Corpus):
        self.reference_corpus = reference_corpus
        self.reference_document_ids = [document.document_id.text for document in self.reference_corpus.iter_documents()]
        self.true_terms = self.extract_true_terms()
        self.predictions = {}

    def extract_true_terms(self):
        true_terms = {}
        for document in self.reference_corpus.iter_documents():
            document_id = document.document_id.text
            terms = [Evaluator.preprocess(term.text) for term in document.terms.term]
            true_terms[document_id] = terms
        return true_terms

    def add_prediction(self, name, prediction):
        if not type(prediction) is dict:
            raise TypeError("prediction must be a dictionary with key=document_id, value=list of terms.")
        if len(self.predictions) > 0:
            sample_document_ids = set(self.reference_document_ids)
            # check all predictions contain the same document ids
            compare_document_ids = set(prediction.keys())
            if sample_document_ids != compare_document_ids:
                difference = set(sample_document_ids - compare_document_ids)
                difference.update(compare_document_ids - sample_document_ids)
                raise IndexError(f"Found difference of Document IDs: {difference}")
        self.predictions[name] = prediction

    @staticmethod
    def preprocess(term):
        return term.lower()

    @staticmethod
    def get_precision(true_terms, pred_terms):
        range_precision = []
        correct_terms = []
        for i, term in enumerate(pred_terms):
            if term in true_terms:
                correct_terms.append(term)
            range_precision.append(len(correct_terms)/(i+1))
        return range_precision

    @staticmethod
    def get_relative_recalls(true_terms, preds):
        if len(preds) == 0:
            raise IndexError("Minimum 1 preds provided.")
        # if the number of preds are different, take the minimum length
        values_list = list(preds.values())
        min_length = min([len(val) for val in values_list])
        range_relative_recalls = {name: [] for name in preds}
        correct_term_pool = set()
        current_correct_terms = {name: [] for name in preds}
        for i in range(min_length):
            for name, terms in preds.items():
                if terms[i] in true_terms:
                    current_correct_terms[name].append(terms[i])
                    correct_term_pool.add(terms[i])
            for name in preds:
                relative_recall = (len(current_correct_terms[name])/len(correct_term_pool)) \
                    if len(correct_term_pool) > 0 else 0
                range_relative_recalls[name].append(relative_recall)
        return range_relative_recalls

    @staticmethod
    def get_average_score(score):
        max_length = max([len(score_vals) for score_vals in score.values()])
        average_score = [0 for _ in range(max_length)]
        # calculate per n according to the number of document that has at least n number of scores
        num_documents_per_length = [0 for _ in range(max_length)]
        for document_id in score:
            for i in range(len(score[document_id])):
                average_score[i] += score[document_id][i]
                num_documents_per_length[i] += 1
        average_score = [average_score[i] / num_documents_per_length[i] for i in range(max_length)]
        return average_score

    @staticmethod
    def get_average_scores(scores):
        average_score = {}
        for name in scores:
            average_score[name] = Evaluator.get_average_score(scores[name])
        return average_score

    @staticmethod
    def visualize_scores(scores, output_file):
        scores_df = {"x": [], "method": [], "score": []}
        for name, score in scores.items():
            for i, s in enumerate(score):
                scores_df["x"].append(i+1)
                scores_df["method"].append(name)
                scores_df["score"].append(s)
        scores_df = pd.DataFrame(scores_df)
        chart = alt.Chart(scores_df).mark_line().encode(x="x", y="score", color="method")
        chart.save(output_file)

    def calculate_precision_all(self):
        all_precision = {}
        for name, prediction in self.predictions.items():
            precision = {}
            for document_id in prediction:
                if document_id not in self.reference_document_ids:
                    raise KeyError(f"Document ID: {document_id} not in Evaluator's reference corpus.")
                true_terms = self.true_terms[document_id]
                pred_terms = prediction[document_id]
                precision[document_id] = Evaluator.get_precision(true_terms, pred_terms)
            all_precision[name] = precision
        return all_precision

    def calculate_relative_recalls_all(self):
        all_relative_recalls = {name: {} for name in self.predictions}
        for document_id in self.reference_document_ids:
            preds = {name: pred[document_id] for name, pred in self.predictions.items()}
            relative_recalls = Evaluator.get_relative_recalls(self.true_terms[document_id], preds)
            for name in self.predictions:
                all_relative_recalls[name][document_id] = relative_recalls[name]
        return all_relative_recalls


if __name__ == "__main__":
    labelled_corpus = Corpus("../data/test/samples_with_terms.xml")
    evaluator = Evaluator(labelled_corpus)
    accurate_preds = {document_id: evaluator.true_terms[document_id][:10] for document_id in evaluator.true_terms}
    evaluator.add_prediction("TEST", accurate_preds)
    evaluator.add_prediction("TEST2", accurate_preds)
    precisions = evaluator.calculate_precision_all()
    avg_precisions = Evaluator.get_average_scores(precisions)
    print(avg_precisions)
    # Evaluator.visualize_scores(avg_precisions, "../plots/precision.html")
    recalls = evaluator.calculate_relative_recalls_all()
    avg_recalls = Evaluator.get_average_scores(recalls)
    print(avg_recalls)
    # Evaluator.visualize_scores(avg_recalls, "../plots/recall.html")
