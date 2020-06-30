import numpy as np
import pandas as pd
import altair as alt
from altair import datum
from kargo import logger
from kargo.corpus import Corpus
log = logger.get_logger(__name__, logger.INFO)


class Evaluator(object):

    def __init__(self, reference_corpus: Corpus):
        self.reference_corpus = reference_corpus
        self.reference_document_ids = [document.document_id.text for document in self.reference_corpus.iter_documents()]
        self.true_terms = self.extract_true_terms()
        self.predictions = {}
        self.scores = {}

    def extract_true_terms(self):
        true_terms = {}
        for document in self.reference_corpus.iter_documents():
            document_id = document.document_id.text
            terms = [Evaluator.preprocess(term.word.text) for term in document.terms.term]
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
    def get_aggr_score(score):
        max_length = max([len(score_vals) for score_vals in score.values()])
        accumulative_score = [[] for _ in range(max_length)]
        # calculate per n according to the number of document that has at least n number of scores
        for document_id in score:
            for i in range(len(score[document_id])):
                accumulative_score[i].append(score[document_id][i])
        aggr_score = [
            (np.mean(accumulative_score[i]), np.std(accumulative_score[i]))
            for i in range(max_length) if len(accumulative_score[i]) > 0
        ]
        return aggr_score

    @staticmethod
    def get_aggr_scores_df(scores, score_name):
        scores_dict = {"method": [], "k": [], score_name: [], "std": [], "ymin": [], "ymax": []}
        for method_name, score in scores.items():
            aggr_score = Evaluator.get_aggr_score(score)
            for k, (score_mean, score_std) in enumerate(aggr_score):
                scores_dict["method"].append(method_name)
                scores_dict["k"].append(k+1)
                scores_dict[score_name].append(score_mean)
                scores_dict["std"].append(score_std)
                scores_dict["ymin"].append(score_mean-2*score_std)
                scores_dict["ymax"].append(score_mean+2*score_std)
        scores_df = pd.pivot_table(
            pd.DataFrame(scores_dict),
            values=[score_name, "ymin", "ymax", "std"], index=["method", "k"]
        ).reset_index()
        return scores_df

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

    def calculate_fscores_all(self, precisions, relative_recalls):
        f_scores = {}
        for method_name in self.predictions:
            for doc_id in self.reference_document_ids:
                for i in range(len(relative_recalls[method_name][doc_id])):
                    p = precisions[method_name][doc_id][i]
                    r = relative_recalls[method_name][doc_id][i]
                    f = 0 if p == r == 0 else (2 * p * r) / (p + r)
                    if method_name not in f_scores:
                        f_scores[method_name] = {}
                    if doc_id not in f_scores[method_name]:
                        f_scores[method_name][doc_id] = []
                    f_scores[method_name][doc_id].append(f)
        return f_scores

    def evaluate_and_visualize(self, output_file):
        # Evaluate precisions
        precisions = self.calculate_precision_all()
        precisions_df = Evaluator.get_aggr_scores_df(precisions, "precisions")
        # Evaluate recalls
        relative_recalls = self.calculate_relative_recalls_all()
        relative_recalls_df = Evaluator.get_aggr_scores_df(relative_recalls, "relative recalls")
        # Calculate F-score
        f_scores = self.calculate_fscores_all(precisions, relative_recalls)
        f_scores_df = Evaluator.get_aggr_scores_df(f_scores, "F-score")
        # Combine scores
        combine_df = f_scores_df[["method", "k", "F-score"]].merge(
            precisions_df[["method", "k", "precisions"]],
            how="inner",
            on=["method", "k"]
        ).merge(
            relative_recalls_df[["method", "k", "relative recalls"]],
            how="inner",
            on=["method", "k"]
        )
        combine_df = combine_df.rename(columns={
            "precisions": "Precisions",
            "relative recalls": "Relative Recalls"
        })
        combine_melt_df = pd.melt(
            combine_df, id_vars=["method", "k"], value_vars=["F-score", "Precisions", "Relative Recalls"]
        )
        combine_melt_df.columns = ["Method", "k", "Evaluation", "Score"]
        click = alt.selection_multi(fields=["Method"])
        method_charts = alt.Chart(combine_melt_df).mark_line(point=True).encode(
            x="k",
            y="Score",
            color="Method",
            column="Evaluation",
            tooltip=["Method", "k", "Score"]
        ).transform_filter(
            click
        )
        overall_chart = alt.Chart(combine_melt_df).mark_bar().encode(
            x=alt.X("mean(Score):Q", title="Avg F-score"),
            y=alt.Y("Method", sort="-x"),
            color=alt.condition(click, "Method", alt.value("lightgray"))
        ).transform_filter(
            datum.Evaluation == "F-score"
        ).properties(
            selection=click
        )
        # f10_chart = alt.Chart(combine_melt_df).mark_bar().encode(
        #     # x=alt.X("mean(Score):Q", title="Avg F-score"),
        #     x=alt.X("Score:Q", title="F@10"),
        #     y=alt.Y("Method", sort="-x"),
        #     # color=alt.condition(click, "Method", alt.value("lightgray"))
        #     color=alt.Color("Method:N", legend=None)
        # ).transform_filter(
        #     (datum.Evaluation == "F1-score") & (datum.k == 10)
        # )
        all_charts = method_charts & overall_chart
        all_charts = all_charts.configure_title(
            fontSize=24,
            font="Times New Roman"
        ).configure_header(
            titleFont="Times New Roman",
            titleFontSize=22,
            labelFont="Times New Roman",
            labelFontSize=20
        ).configure_axis(
            titleFont="Times New Roman",
            titleFontSize=18,
            labelFont="Times New Roman",
            labelFontSize=18
        ).configure_legend(
            titleFont="Times New Roman",
            titleFontSize=20,
            labelFont="Times New Roman",
            labelFontSize=18
        )
        all_charts.save(output_file)


if __name__ == "__main__":
    labelled_corpus = Corpus("../data/test/samples_with_terms.xml")
    evaluator = Evaluator(labelled_corpus)
    accurate_preds = {document_id: evaluator.true_terms[document_id][:10] for document_id in evaluator.true_terms}
    evaluator.add_prediction("TEST", accurate_preds)
    evaluator.add_prediction("TEST2", accurate_preds)
    evaluator.evaluate_and_visualize("../data/test/evaluation/eval.html")
