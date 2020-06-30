import csv
import json
import os
import pickle
from hashlib import md5
from tqdm import tqdm
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from corpus import Corpus


class CustomVectorizer(TfidfVectorizer):
    def __init__(self, **kwargs):
        super(CustomVectorizer, self).__init__(**kwargs)
        self.stemmer = PorterStemmer()
        self.stop_words = ENGLISH_STOP_WORDS.union(SPACY_STOP_WORDS)
        self.input = "filename"
        self.max_df = 0.75
        self.min_df = 20

    def build_analyzer(self):
        analyzer = super(CustomVectorizer, self).build_analyzer()
        return lambda doc: (self.stemmer.stem(w) for w in analyzer(doc))


class TopicModeler(object):

    def __init__(self, corpus, text_folder, models_folder, override):
        self.corpus = corpus
        self.text_folder = text_folder
        self.models_folder = models_folder
        self.override = override
        self.document_ids = []

    def start(self, n_topics):
        self.preprocess()
        text_model, text_bow = self.feature_extraction()
        text_lda = self.run_lda(text_bow, text_model, n_topics, n_top_words=10)

    def preprocess(self):
        existing_files = [fname for fname in os.listdir(self.text_folder) if fname.endswith("str")]
        for document in tqdm(self.corpus.iter_documents(), total=len(self.corpus)):
            document_id = document.document_id.text
            if document_id not in existing_files and document_id not in self.document_ids \
                    and document.content.countchildren() > 0:
                text = [p.text for p in document.content.p]
                title_text = document.title.text.lower()
                text_prep = "\n".join([title_text] + [t.lower() for t in text])
                with open(os.path.join(self.text_folder, f"{document_id}.txt"), "w") as fout:
                    fout.writelines(text_prep)

    def feature_extraction(self):
        vocab_filename = "vocab_10_tfidf.pkl"
        self.document_ids = [fname.split(".")[0] for fname in os.listdir(self.text_folder) if fname.endswith(".txt")]
        filenames = [os.path.join(self.text_folder, f"{fname}.txt") for fname in self.document_ids]
        bow_model = CustomVectorizer()
        bow = bow_model.fit_transform(filenames)
        print(f"Number of features: {len(bow_model.get_feature_names())}")
        if len(self.document_ids) != bow.shape[0]: raise ValueError("Check document length != features")
        with open( os.path.join( self.models_folder, vocab_filename ), "wb" ) as fout:
            pickle.dump(bow_model.vocabulary_, fout)
        return bow_model, bow

    def run_lda(self, features, vect_model, n_topics, n_top_words):
        lda_model_filename = "lda_model_10_tfidf.pkl"
        topic_filename = "topic_10_tfidf.csv"
        if self.override or lda_model_filename not in os.listdir(self.models_folder):
            lda_model = LDA(n_components=n_topics)
            lda_model.fit(features)
            with open(os.path.join(self.models_folder, lda_model_filename), "wb") as fout:
                pickle.dump(lda_model, fout)
        else:
            with open(os.path.join(self.models_folder, lda_model_filename), "r") as fin:
                lda_model = pickle.load(fin)
        words = vect_model.get_feature_names()
        for topic_idx, topic in enumerate(lda_model.components_):
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        text_topics = lda_model.transform(features).argmax(axis=1).ravel()
        text_probs = lda_model.transform(features).max(axis=1).ravel()
        with open(os.path.join(self.models_folder, topic_filename), "w") as fout:
            csv_writer = csv.DictWriter(fout, fieldnames=["document_id", "topic_id", "topic_prob"])
            csv_writer.writeheader()
            for i in range(len(self.document_ids)):
                csv_writer.writerow({
                    "document_id": self.document_ids[i],
                    "topic_id": text_topics[i],
                    "topic_prob": text_probs[i]
                })
        return lda_model

    def get_top_news_from_topic(self, topic_csv, topic_id, threshold=0.9):
        with open(os.path.join(self.models_folder, topic_csv), "r") as fin:
            topic_mappings = csv.DictReader(fin)
            valid_ids = [
                topic_map["document_id"] for topic_map in topic_mappings
                if int(topic_map["topic_id"]) == topic_id and float(topic_map["topic_prob"]) > threshold
            ]
        # sorted(valid_docs, key=lambda d: d["topic_prob"])
        # top_document_ids = [valid_doc["document_id"] for valid_doc in valid_docs[-top_n:]]
        new_corpus = Corpus()
        for document in self.corpus.iter_documents():
            if document.document_id in valid_ids:
                new_corpus.add_document_from_element(document)
        return new_corpus


def produce_jsonl_with_existing(existing_jsonl, corpus, output_jsonl, n_samples=50):
    existing_document_ids = []
    with open(existing_jsonl, "r") as fin:
        json_lines = fin.readlines()
        for json_line in json_lines:
            line = json.loads(json_line)
            document_id = md5(line["text"].split("|")[0].encode("utf-8")).hexdigest()[-6:]
            existing_document_ids.append(document_id)
    out_jsons = []
    for document in corpus.iter_documents():
        if document.document_id not in existing_document_ids:
            text = "|".join([document.title.text] + [p.text for p in document.content.p])
            out_jsons.append({"text": text})
            if len(out_jsons) == n_samples: break
    with open(output_jsonl, "w") as fout:
        for out_json in out_jsons:
            fout.writelines(json.dumps(out_json) + "\n")


if __name__ == "__main__":
    corpus = Corpus("../data/interim/all-v2.xml")
    topic_modeler = TopicModeler(corpus, "../data/processed/txts/", "../data/interim/topic_models", override=True)
    # topic_modeler.start(n_topics=10)
    topic_corpus = topic_modeler.get_top_news_from_topic("topic_10_tfidf.csv", 3)
    topic_corpus.write_xml_to("../data/processed/lda_sampling_10p.xml")
    # lda_corpus = Corpus("../data/processed/lda_sampling_10p.xml")
    # produce_jsonl_with_existing(
    #     "../data/manual/annotation.json1",
    #     lda_corpus,
    #     "../data/manual/lda_sample_p10_1.json1"
    # )

