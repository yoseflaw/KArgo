from csv import DictReader
import json

import stanza
from tqdm import tqdm
from anytree import Node, RenderTree
from corpus import StanfordCoreNLPCorpus
from nltk.corpus import wordnet as wn


class SentenceTree(object):
    valid_attrs = ["id", "word", "lemma", "pos", "deprel", "head_id", "head_text", "ner", "children"]

    def __init__(self, sentence):
        self.sentence = sentence
        self.children = {}
        self.tokens = {
            "0": {
                "id": "0",
                "word": "root",
                "lemma": "root",
                "pos": "",
                "deprel": "",
                "head_id": "",
                "head_text": "",
                "ner": ""
            }
        }
        self.sequence = []
        for token in sentence.tokens.token:
            if token.deprel_head_id.text in self.tokens:
                if "children" in self.tokens[token.deprel_head_id.text]:
                    self.tokens[token.deprel_head_id.text]["children"].append(token.get("id"))
                else:
                    self.tokens[token.deprel_head_id.text]["children"] = [token.get("id")]
            else:
                self.tokens[token.deprel_head_id.text] = {"children": [token.get("id")]}
            token_map = {
                "id": token.get("id"),
                "word": token.word.text.lower(),
                "lemma": token.lemma.text,
                "pos": token.POS.text,
                "deprel": token.deprel.text,
                "head_id": token.deprel_head_id.text,
                "head_text": token.deprel_head_text.text.lower(),
                "ner": token.ner.text
            }
            if token.get("id") in self.tokens and "children" in self.tokens[token.get("id")]:
                token_map["children"] = self.tokens[token.get("id")]["children"]
            self.tokens[token.get("id")] = token_map
            self.sequence.append(token.get("id"))

    def get_token_key(self, token_id):
        return token_id, self.tokens[token_id]["word"]

    def get_token_key_str(self, token_id):
        return "|".join(self.get_token_key(token_id))

    def get_seq_key(self, i):
        return self.get_token_key(self.sequence[i])

    def get_seq_key_str(self, i):
        return "|".join(self.get_seq_key(i))

    def get_token_attr(self, token_id, attr):
        if attr not in self.valid_attrs: return None
        return self.tokens[token_id][attr]

    def get_seq_attr(self, i, attr):
        if attr not in self.valid_attrs: return None
        return self.get_token_attr(self.sequence[i], attr)

    def get_list(self, attr):
        return [self.get_token_attr(token_id, attr) for token_id in self.sequence]

    def find_closest_parent(self, head_key, candidate_head_keys):
        if head_key == self.get_token_key("0") and head_key not in candidate_head_keys:
            return None, None
        elif head_key in candidate_head_keys:
            return head_key
        else:
            next_head_id = self.get_token_attr(head_key[0], "head_id")
            return self.find_closest_parent(self.get_token_key(next_head_id), candidate_head_keys)

    def get_dependency_relations(self, phrase, start_index):
        all_children = {}
        for i in range(start_index, start_index+len(phrase)):
            curr_head_id = self.get_seq_attr(i, "head_id")
            if curr_head_id not in self.sequence[start_index:start_index+len(phrase)]:
                head_key = self.get_token_key(curr_head_id)
                for child_id in self.get_token_attr(head_key[0], "children"):
                    curr_child_key = self.get_token_key(child_id)
                    wn_synset = wn.synsets(head_key[1])
                    parent_wn_lemma = wn_synset[0].lemmas()[0].name() if wn_synset else head_key[1]
                    all_children[curr_child_key] = {
                        "extracted_word": self.get_seq_key(i),
                        "extracted_word_deprel": self.get_seq_attr(i, "deprel"),
                        "parent": head_key,
                        "parent_pos": self.get_token_attr(curr_head_id, "pos"),
                        "parent_wn_lemma": parent_wn_lemma,
                        "entity": [],
                        "head_entity_deprel": None
                    }
        for i, curr_head_id in enumerate(self.get_list("head_id")):
            curr_token_key = self.get_seq_key(i)
            curr_head_key = self.get_token_key(curr_head_id)
            closest_parent = self.find_closest_parent(curr_head_key, all_children.keys())
            if curr_token_key in all_children:
                all_children[curr_token_key]["head_entity_deprel"] = self.get_seq_attr(i, "deprel")
                # all_children[curr_token_key]["entity"].append(self.get_seq_key_str(i))
                all_children[curr_token_key]["entity"].append(self.get_seq_attr(i, "word"))
            elif closest_parent[0]:
                all_children[closest_parent]["entity"].append(self.get_seq_attr(i, "word"))
        return all_children


class RelationExtractor(object):

    def __init__(self, snlp_corpus, extracted_terms_path):
        self.corpus = snlp_corpus
        self.terms = self.read_extracted_terms(extracted_terms_path)

    def read_extracted_terms(self, filepath):
        terms = {}
        with open(filepath, "r") as f:
            reader = DictReader(f)
            for row in reader:
                terms[row["document_id"]] = row["terms"].split("|")
        return terms

    def extract(self):
        relations = {}
        for document in tqdm(self.corpus.iter_documents(), total=len(self.corpus)):
            doc_id = document.get("id")
            tokenized_terms = [term.split() for term in self.terms[doc_id]]
            for sentence in document.sentences.sentence:
                sentence_tree = SentenceTree(sentence)
                sentence_words = sentence_tree.get_list("word")
                for extracted_tokens in tokenized_terms:
                    for i in range(len(sentence_words)-len(extracted_tokens)):
                        if sentence_words[i:i+len(extracted_tokens)] != extracted_tokens: continue
                        candidates = sentence_tree.get_dependency_relations(extracted_tokens, i)
                        candidates = self.filter(candidates)
                        for entity in candidates:
                            if entity == candidates[entity]["extracted_word"]: continue
                            parent_id, parent_word = candidates[entity]["parent"]
                            term = " ".join(extracted_tokens)
                            potential_entity = " ".join(candidates[entity]["entity"])
                            #relation = candidates[entity]["parent_wn_lemma"] + "(" + \
                            # relation = sentence_tree.get_token_attr(parent_id, "lemma") + "(" + \
                            #            candidates[entity]["extracted_word_deprel"] + "," + \
                            #            candidates[entity]["head_entity_deprel"] + ")"
                            relation = sentence_tree.get_token_attr(parent_id, "lemma")
                            actors = {
                                candidates[entity]["extracted_word_deprel"]: term,
                                candidates[entity]["head_entity_deprel"]: potential_entity,
                                "sentence": " ".join(sentence_words)
                            }
                            if relation in relations:
                                relations[relation].append(actors)
                            else:
                                relations[relation] = [actors]
                            # term = " ".join(extracted_tokens)
                            # node_map = {term: Node(term)}
                            # for head_key in potential_entities:
                            #     parent = potential_entities[head_key]["parent"]
                            #     if parent not in node_map:
                            #         node_map[parent] = Node(sentence_tree.get_token_key_str(parent[0]), node_map[term])
                            #     inferred_entity = " ".join(potential_entities[head_key]["entity"])
                            #     node_map[inferred_entity] = Node(inferred_entity, node_map[parent])
                            # for pre, fill, node in RenderTree(node_map[term]):
                            #     print(f"{pre}{node.name}")
        return relations

    def filter(self, candidates):
        excluded_deprels = ("det", "aux", "case", "mark", "punct", "cop", "advmod", "aux:pass", "cc")
        filtered_candidates = {}
        for entity in candidates:
            if candidates[entity]["head_entity_deprel"] not in excluded_deprels and candidates[entity]["parent_pos"] and candidates[entity]["parent_pos"][0] == "V":
                filtered_candidates[entity] = candidates[entity]
        return filtered_candidates


if __name__ == "__main__":
    snlp_folder = "../data/test/core_nlp_samples"
    snlp_corpus = StanfordCoreNLPCorpus(snlp_folder)
    relator = RelationExtractor(snlp_corpus, "../data/test/extracted_terms_sample/mprank.csv")
    candidate_relations = relator.extract()
    with open("../data/test/extracted_relations_sample/deprel.json", "w") as json_file:
        json.dump(candidate_relations, json_file, indent=2)
    # sample_text = "Alitalia has strengthened its partnership with CSafe Global by approving the use of RAP active temperature-controlled containers"
    # stanza_nlp_w_ssplit = stanza.Pipeline(
    #     "en",
    #     processors={"tokenize": "lines", "ner": "default", "lemma": "lines", "pos": "gum", "depparse": "lines"},
    #     verbose=False
    # )
    # doc = stanza_nlp_w_ssplit(sample_text)
    # for sentence in doc.sentences:
    #     for token in sentence.tokens:
    #         print(token)
