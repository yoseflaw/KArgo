from collections import OrderedDict
import json
from csv import DictReader

import opennre
from tqdm import tqdm
from corpus import StanfordCoreNLPCorpus
import Levenshtein as Lev
import numpy as np
from sklearn.cluster import DBSCAN


class Token(object):

    def __init__(self, token_id, word, lemma, offset_begin, offset_end, pos, deprel, head_id, head_text, ner):
        self.token_id = token_id
        self.word = word
        self.lemma = lemma
        self.offset_begin = offset_begin
        self.offset_end = offset_end
        self.pos = pos
        self.deprel = deprel
        self.head_id = head_id
        self.head_text = head_text
        self.ner = ner

    def __str__(self):
        return self.word

    def __repr__(self):
        return f"Token('{self.token_id}', '{self.word}', '{self.pos}', '{self.ner}')"


class SentenceParser(object):
    valid_attrs = [
        "id", "word", "lemma", "offset_begin", "offset_end", "pos", "deprel", "head_id", "head_text", "ner"
    ]

    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_offset = None
        self.tokens = OrderedDict()
        for token in sentence.tokens.token:
            if token.get("id") == "1":
                self.sentence_offset = int(token.CharacterOffsetBegin.text)
            token_map = {
                "token_id": token.get("id"),
                "word": token.word.text.lower(),
                "lemma": token.lemma.text,
                "offset_begin": int(token.CharacterOffsetBegin.text) - self.sentence_offset,
                "offset_end": int(token.CharacterOffsetEnd.text) - self.sentence_offset,
                "pos": token.POS.text,
                "deprel": token.deprel.text,
                "head_id": token.deprel_head_id.text,
                "head_text": token.deprel_head_text.text.lower(),
                "ner": token.ner.text
            }
            self.tokens[token.get("id")] = Token(**token_map)

    def __str__(self):
        str_rep = []
        current_offset = 0
        for token_id, token in self.tokens.items():
            while current_offset < token.offset_begin:
                str_rep.append(" ")
                current_offset += 1
            str_rep.append(token.word)
            current_offset = token.offset_end
        return "".join(str_rep)

    def get_list(self, attr):
        attr_list = []
        for token_id, token in self.tokens.items():
            attr_list.append(getattr(token, attr))
        return attr_list

    def get_token_attr(self, token_id, attr):
        if attr not in self.valid_attrs: return None
        return getattr(self.tokens[token_id], attr)

    # Need to update this
    def get_entities(self):
        ents = []
        ent = []
        for token_id, token in self.tokens.items():
            if token.ner[0] in ("B", "S"):
                ent = [token]
            elif token.ner[0] in ("I", "E"):
                ent.append(token)
            if token.ner[0] in ("E", "S") or (token.ner[0] in ("B", "I") and int(token_id) == len(self.tokens)):
                ents.append(ent)
        return ents

    # This will only return the first occurrence of a term in the sentence
    def is_term_exist(self, term_words):
        sentence_words = self.get_list("word")
        for i in range(len(sentence_words)-len(term_words)):
            if sentence_words[i:i+len(term_words)] == term_words:
                term_tokens = []
                for j in range(i, i+len(term_words)):
                    term_tokens.append(self.tokens[str(j+1)])
                return term_tokens
        return None

    def get_terms_exist(self, terms_words):
        exist = []
        for term_words in terms_words:
            tokens = self.is_term_exist(term_words)
            if tokens: exist.append(tokens)
        return exist

    def get_tokens_subset(self, begin_id, begin_end):
        subset_tokens = []
        for i in range(begin_id, begin_end):
            subset_tokens.append(self.tokens[str(i)])
        return subset_tokens


class RelationExtractor(object):

    def __init__(self, window_size, include_ne, closest_term_only):
        self.window_size = window_size
        self.include_ne = include_ne
        self.closest_term_only = closest_term_only

    @staticmethod
    def read_extracted_terms(extracted_terms_path):
        terms = {}
        with open(extracted_terms_path, "r") as f:
            reader = DictReader(f)
            for row in reader:
                terms[row["document_id"]] = row["terms"].split("|")
        return terms

    @staticmethod
    def reduce_duplicate_entities(entity_tokens):
        unique_entities = []
        for i in range(len(entity_tokens)):
            ent1 = entity_tokens[i]
            ent1_set = set(range(
                int(ent1[0].token_id), int(ent1[-1].token_id)+1
            ))
            found = False
            for j in range(len(unique_entities)):
                ent2 = unique_entities[j]
                ent2_set = set(range(
                    int(ent2[0].token_id), int(ent2[-1].token_id)+1
                ))
                found = len(ent1_set & ent2_set) > 0
                if found: break
            if not found:
                unique_entities.append(ent1)
        return unique_entities

    @staticmethod
    def write_relations_to(relations, output_file):
        with open(output_file, "w") as fout:
            json.dump(relations, fout, indent=2)

    def get_terms_occurrence(self, parsed_sentence, tokenized_terms, extract_tokens, n_outer_tokens):
        sentence_terms = parsed_sentence.get_terms_exist(tokenized_terms)
        if self.include_ne:
            sentence_terms += parsed_sentence.get_entities()
        entities = RelationExtractor.reduce_duplicate_entities(sentence_terms)
        if len(entities) < 2: return []
        sorted_entities = sorted(entities, key=lambda e: int(e[0].token_id))
        cooccurs = []
        for i in range(len(sorted_entities)):
            head_tokens = sorted_entities[i]
            head_end = int(head_tokens[-1].token_id)
            considered_tail = min(i+2, len(sorted_entities)) if self.closest_term_only else len(sorted_entities)
            for j in range(i+1, considered_tail):
                tail_tokens = sorted_entities[j]
                tail_begin = int(tail_tokens[0].token_id)
                if tail_begin - head_end <= self.window_size:
                    cooccur = {
                        "text": str(parsed_sentence),
                        "head": head_tokens,
                        "tail": tail_tokens,
                    }
                    if extract_tokens:
                        cooccur.update({
                            "in_between": parsed_sentence.get_tokens_subset(head_end + 1, tail_begin)
                        })
                        if n_outer_tokens:
                            head_begin = int(head_tokens[0].token_id)
                            tail_end = int(tail_tokens[-1].token_id)
                            cooccur.update({
                                "prefix": parsed_sentence.get_tokens_subset(
                                    max(1, head_begin - n_outer_tokens),
                                    head_begin
                                ),
                                "suffix": parsed_sentence.get_tokens_subset(
                                    tail_end+1,
                                    min(len(parsed_sentence.tokens), tail_end + 1 + n_outer_tokens)
                                )
                            })
                    cooccurs.append(cooccur)
        return cooccurs

    def get_all_cooccurrences(self, snlp_corpus, extracted_terms_path, extract_tokens=False, n_outer_tokens=0):
        all_cooccurrences = []
        terms = RelationExtractor.read_extracted_terms(extracted_terms_path)
        for document in tqdm(snlp_corpus.iter_documents(), total=len(snlp_corpus), disable=True):
            doc_id = document.get("id")
            tokenized_terms = [term.split() for term in terms[doc_id]]
            for sentence in document.sentences.sentence:
                parsed_sentence = SentenceParser(sentence)
                cooccurrences = self.get_terms_occurrence(
                    parsed_sentence, tokenized_terms, extract_tokens, n_outer_tokens
                )
                all_cooccurrences += cooccurrences
        return all_cooccurrences


class ClusteringRE(RelationExtractor):

    def __init__(self, n_outer_tokens, generalize, clusterer_params, window_size, closest_term_only, include_ne):
        super().__init__(window_size, include_ne, closest_term_only)
        self.n_outer_tokens = n_outer_tokens
        self.patterns = ["in_between"] if not n_outer_tokens else ["in_between", "prefix", "suffix"]
        self.generalize = generalize if generalize in ("word", "lemma", "pos") else "word"
        self.clusterer_params = clusterer_params

    def calc_dist_matrix(self, cooccurrences):
        distance_matrix = np.zeros((len(self.patterns), len(cooccurrences), len(cooccurrences)))
        generalized_patterns = []
        for i in range(len(cooccurrences)):
            generalized_pattern = {}
            for pattern in self.patterns:
                generalized_pattern[pattern] = [
                    getattr(token, self.generalize) for token in cooccurrences[i][pattern]
                ]
            generalized_patterns.append(generalized_pattern)
        for p, pattern in enumerate(self.patterns):
            for i in tqdm(range(len(generalized_patterns))):
                pattern_i = generalized_patterns[i][pattern]
                for j in range(i+1, len(generalized_patterns)):
                    pattern_j = generalized_patterns[j][pattern]
                    dist = 1 - Lev.seqratio(pattern_i, pattern_j)
                    distance_matrix[p, i, j] = distance_matrix[p, j, i] = dist
        distance_matrix = np.mean(distance_matrix, axis=0)
        return distance_matrix

    def cluster(self, cooccurrences):
        distance_matrix = self.calc_dist_matrix(cooccurrences)
        clusterer = DBSCAN(**self.clusterer_params)
        clusters = clusterer.fit_predict(distance_matrix)
        relations = {}
        for i, cooccurrence in enumerate(cooccurrences):
            cluster_id = str(clusters[i])
            rel_elmt = {
                "text": cooccurrence["text"],
                "head_words": cooccurrence["text"][
                              cooccurrence["head"][0].offset_begin:
                              cooccurrence["head"][-1].offset_end
                              ],
                "tail_words": cooccurrence["text"][
                              cooccurrence["tail"][0].offset_begin:
                              cooccurrence["tail"][-1].offset_end
                              ]
            }
            for pattern in self.patterns:
                if len(cooccurrence[pattern]) > 0:
                    rel_elmt[f"{pattern}_words"] = cooccurrence["text"][
                                                   cooccurrence[pattern][0].offset_begin:
                                                   cooccurrence[pattern][-1].offset_end
                                                   ]
                else:
                    rel_elmt[f"{pattern}_words"] = ""
            if cluster_id in relations:
                relations[cluster_id].append(rel_elmt)
            else:
                relations[cluster_id] = [rel_elmt]
        return relations

    def extract(self, snlp_corpus, extracted_terms_path, output_file):
        all_cooccurrences = self.get_all_cooccurrences(
            snlp_corpus, extracted_terms_path, extract_tokens=True, n_outer_tokens=self.n_outer_tokens
        )
        relations = self.cluster(all_cooccurrences)
        if output_file: RelationExtractor.write_relations_to(relations, output_file)
        return relations


class TransferRE(RelationExtractor):

    def __init__(self, model_name, prob_threshold, window_size, include_ne, closest_term_only):
        super().__init__(window_size, include_ne, closest_term_only)
        self.model = opennre.get_model(model_name)
        self.prob_threshold = prob_threshold

    def infer(self, cooccurrences):
        relations = {}
        for cooccurrence in cooccurrences:
            relation, prob = self.model.infer({
                "text": cooccurrence["text"],
                "h": {
                    "pos": (cooccurrence["head"][0].offset_begin, cooccurrence["head"][-1].offset_end)
                },
                "t": {
                    "pos": (cooccurrence["tail"][0].offset_begin, cooccurrence["tail"][-1].offset_end)
                }
            })
            if prob >= self.prob_threshold:
                rel_elmt = {
                    "text": cooccurrence["text"],
                    "head_words": cooccurrence["text"][
                                  cooccurrence["head"][0].offset_begin:cooccurrence["head"][-1].offset_end],
                    "tail_words": cooccurrence["text"][
                                  cooccurrence["tail"][0].offset_begin:cooccurrence["tail"][-1].offset_end],
                    "prob": prob
                }
                if relation in relations:
                    relations[relation].append(rel_elmt)
                else:
                    relations[relation] = [rel_elmt]
        return relations

    def extract(self, snlp_corpus, extracted_terms_path, output_file=None):
        all_cooccurrences = self.get_all_cooccurrences(
            snlp_corpus, extracted_terms_path, extract_tokens=False, n_outer_tokens=0
        )
        relations = self.infer(all_cooccurrences)
        if output_file: RelationExtractor.write_relations_to(relations, output_file)
        return relations


if __name__ == "__main__":
    dbscan_params = {
        "eps": 0.5,
        "min_samples": 3,
        "metric": "precomputed"
    }
    relator = ClusteringRE(
        n_outer_tokens=3,
        generalize="lemma",
        clusterer_params=dbscan_params,
        window_size=10,
        closest_term_only=True,
        include_ne=True
    )
    # relator = TransferRE(
    #     model_name="wiki80_cnn_softmax",
    #     prob_threshold=0,
    #     window_size=10,
    #     closest_term_only=True,
    #     include_ne=True
    # )
    # snlp_folder = "../data/test/core_nlp_samples"
    # snlp_folder = "../data/processed/scnlp_lda_all/"
    snlp_folder = "../data/processed/scnlp_xmls/"
    corpus = StanfordCoreNLPCorpus(snlp_folder)
    results = relator.extract(
        corpus,
        "../results/extracted_terms/kpm.csv",
        "../results/extracted_relations/dbscan.json"
    )
    # results = relator.extract(
    #     corpus,
    #     "../data/test/extracted_terms_sample/mprank.csv",
    #     "../results/extracted_relations/wikicnn_0_10_TT.json"
    # )
