from kargo import logger
from kargo.corpus import Corpus
log = logger.get_logger(__name__, logger.INFO)


class Evaluator(object):

    def __init__(self, reference_corpus: Corpus):
        self.reference_corpus = reference_corpus
        self.reference_document_ids = [document.document_id.text for document in self.reference_corpus.iter_documents()]
        self.true_terms = self.extract_true_terms()

    def extract_true_terms(self):
        true_terms = {}
        for document in self.reference_corpus.iter_documents():
            document_id = document.document_id.text
            terms = [Evaluator.preprocess(term.text) for term in document.terms.term]
            true_terms[document_id] = terms
        return true_terms

    @staticmethod
    def preprocess(term):
        return term.lower()

    @staticmethod
    def get_precision(true_terms, pred_terms):
        correct_terms = [term for term in pred_terms if term in true_terms]
        precision = len(correct_terms) / len(pred_terms)
        return precision

    @staticmethod
    def get_relative_recalls(true_terms, preds):
        correct_term_pool = set()
        correct_pred_pool = [[] for _ in range(len(preds))]
        for i, pred_terms in enumerate(preds):
            current_correct_terms = [term for term in pred_terms if term in true_terms]
            correct_pred_pool[i] = current_correct_terms
            correct_term_pool.update(current_correct_terms)
        relative_recalls = [(len(correct_pred) / len(correct_term_pool)) for correct_pred in correct_pred_pool]
        return relative_recalls

    def calculate_precision_all(self, pred_all_documents):
        all_precision = {}
        for document_id in pred_all_documents:
            if document_id not in self.reference_document_ids:
                raise KeyError(f"Document ID: {document_id} not in Evaluator's reference corpus.")
            true_terms = self.true_terms[document_id]
            pred_terms = pred_all_documents[document_id]
            all_precision[document_id] = Evaluator.get_precision(true_terms, pred_terms)
        return all_precision

    def calculate_relative_recalls_all(self, preds_all_documents):
        all_relative_recalls = {}
        if not type(preds_all_documents) is list:
            raise TypeError("preds_all_documents must be a list of set of predictions.")
        if len(preds_all_documents) == 0:
            raise IndexError("preds_all_documents must contain at least 1 set of predictions.")
        sample_document_ids = set(preds_all_documents[0].keys())
        if len(preds_all_documents) > 1:
            # check all predictions contain the same document ids
            for i, pred_all_documents in enumerate(preds_all_documents[1:]):
                compare_document_ids = set(pred_all_documents.keys())
                if sample_document_ids != compare_document_ids:
                    difference = set(sample_document_ids - compare_document_ids)
                    difference.update(compare_document_ids - sample_document_ids)
                    raise IndexError(f"Found difference of Document IDs when comparing index 0 and {i+1}: {difference}")
        for document_id in sample_document_ids:
            preds = [pred[document_id] for pred in preds_all_documents]
            all_relative_recalls[document_id] = Evaluator.get_relative_recalls(
                self.true_terms[document_id], preds
            )
        return all_relative_recalls


if __name__ == "__main__":
    labelled_corpus = Corpus("../data/test/samples_with_terms.xml")
    evaluator = Evaluator(labelled_corpus)
    accurate_preds = evaluator.true_terms
    all_predictions = evaluator.calculate_precision_all(accurate_preds)
    print(all_predictions)
    all_relative_recalls = evaluator.calculate_relative_recalls_all([accurate_preds, accurate_preds])
    print(all_relative_recalls)
