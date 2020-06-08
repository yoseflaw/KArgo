from csv import DictWriter, DictReader
from hashlib import md5
import os
import random
import json
from collections import defaultdict
import html
from bisect import bisect, insort
from tqdm import tqdm
from lxml.objectify import Element
from lxml import etree, objectify
import stanza
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from kargo import logger
log = logger.get_logger(__name__, logger.INFO)


class XMLBase(object):

    def __init__(self, root_name, document_tag):
        self.root_name = root_name
        self.document_tag = document_tag

    def get_root(self):
        return getattr(self, self.root_name)

    def __len__(self):
        return self.get_root().countchildren()

    def __getitem__(self, i):
        return getattr(self.get_root(), self.document_tag)[i]

    def iter_documents(self):
        return self.get_root().iterchildren()

    def get_xml(self):
        objectify.deannotate(getattr(self, self.root_name), xsi_nil=True)
        etree.cleanup_namespaces(getattr(self, self.root_name))
        return etree.tostring(getattr(self, self.root_name), pretty_print=True, encoding="unicode")

    def write_xml_to(self, out_path):
        with open(out_path, "w") as xml_out:
            xml_out.write(self.get_xml())


class AnnotationJSON(object):

    def __init__(self, annotation_file):
        self.documents = {}
        self.irrelevants = []
        with open(annotation_file, "r") as fin:
            anns = fin.readlines()
        for ann in anns:
            doc = json.loads(ann)
            title = doc["text"].split("|")[0]
            text = doc["text"]
            if "meta" in doc and "doc_id" in doc["meta"]:
                doc_id = doc["meta"]["doc_id"]
            else:
                doc_id = md5(title.encode("utf-8")).hexdigest()[-6:]
            has_irrelevant = False
            annotation_mapping = {}
            for tag in doc["labels"]:
                begin, end, term_type = tag
                if term_type == "IRRELEVANT":
                    has_irrelevant = True
                    break
                term = text[begin:end]
                if term in annotation_mapping:
                    annotation_mapping[term].append((begin, end))
                else:
                    annotation_mapping[term] = [(begin, end)]
            if has_irrelevant:
                self.irrelevants.append(doc_id)
            else:
                self.documents[doc_id] = annotation_mapping


class Corpus(XMLBase):
    annotation_regex = r"\[\[(.+?)\]\]"

    def __init__(self, xml_input=None, annotations=None):
        super().__init__("corpus", "document")
        self.corpus = Element("corpus")
        self.url_indices = []
        self.has_terms_locations = False
        self.annotations = annotations.documents if annotations else None
        if xml_input:
            if xml_input and not os.path.exists(xml_input):
                raise FileNotFoundError(f"{xml_input} not found. Check the path again.")
            elif os.path.isfile(xml_input):
                self.read_from_xml(xml_input)
            else:
                self.read_from_folder(xml_input)

    @staticmethod
    def unicodify(text):
        return text.replace("“", "\"")\
            .replace("”", "\"")\
            .replace("’", "'")\
            .replace("‘", "'") \
            .replace("\n", " ")

    def add_document(self, url, title, categories, published_time, content, author=None, topics=None, links=None,
                     terms=None, document_id=None):
        if url is None or len(url) == 0:
            raise KeyError("'url' is mandatory")
        elif url in self.url_indices:
            log.info(f"Ignoring duplicate URL={url}")
            return
        new_document = Element("document")
        title = Corpus.unicodify(title)
        new_document.document_id = md5(title.encode("utf-8")).hexdigest()[-6:] if document_id is None or \
            len(document_id) == 0 else document_id
        new_document.url = url
        new_document.title = title
        new_document.author = author
        new_document.published_time = published_time
        # handle lists
        new_document.categories = Element("categories")
        if categories: new_document.categories.category = categories
        new_document.topics = Element("topics")
        if topics: new_document.topics.topic = topics
        new_document.links = Element("links")
        if links: new_document.links.link = links
        new_document.content = Element("content")
        if content:
            new_document.content.p = [Corpus.unicodify(p) for p in content if p]
        # handle terms
        new_document.terms = Element("terms")
        terms_list = []
        if terms:
            for term in terms:
                term_elmt = Element("term")
                term_elmt.word = term
                term_elmt.locations = Element("locations")
                locations_list = []
                for location in terms[term]:
                    location_elmt = Element("location")
                    location_elmt.begin, location_elmt.end = location
                    locations_list.append(location_elmt)
                term_elmt.locations.location = locations_list
                terms_list.append(term_elmt)
            new_document.terms.term = terms_list
        self.corpus.append(new_document)
        self.url_indices.append(url)

    def add_document_from_element(self, document_elmt):
        # construct terms
        terms_list = {}
        if document_elmt.terms.countchildren() > 0:
            for term in document_elmt.terms.term:
                if term.locations.countchildren() > 0:
                    terms_list[term.word.text] = [(loc.begin.text, loc.end.text) for loc in term.locations.location]
        self.add_document(
            document_elmt.url.text,
            document_elmt.title.text,
            [category.text for category in document_elmt.categories.category
             ] if document_elmt.categories.countchildren() > 0 else None,
            document_elmt.published_time.text,
            [p.text for p in document_elmt.content.p] if document_elmt.content.countchildren() > 0 else None,
            document_elmt.author.text,
            [topic.text for topic in document_elmt.topics.topic] if document_elmt.topics.countchildren() > 0 else None,
            [link.text for link in document_elmt.links.link] if document_elmt.links.countchildren() > 0 else None,
            terms_list if len(terms_list) > 0 else None,
            document_elmt.document_id,
        )

    def filter_empty(self):
        empty_document_list = []
        for document in self.iter_documents():
            if document.content.countchildren() == 0:
                empty_document_list.append(document)
        for document in empty_document_list:
            self.get_root().remove(document)
        return self

    def read_from_xml(self, input_path):
        composites = ["terms", "topics", "content", "links", "categories"]
        corpus_etree = etree.parse(input_path)
        corpus_root = corpus_etree.getroot()
        for document in corpus_root:
            new_document_attrs = {}
            annotated_terms = {}
            contain_terms_elmt = False
            for document_elmt in document:
                if document_elmt.tag == "category":
                    new_document_attrs["categories"] = document_elmt.text.split(";") if document_elmt.text else []
                elif document_elmt.tag == "terms":  # the document has existing annotations
                    for term_elmt in document_elmt:
                        word = None
                        locations = []
                        for item_elmt in term_elmt:
                            if item_elmt.tag == "word":
                                word = item_elmt.text
                            elif item_elmt.tag == "locations":
                                begin, end = None, None
                                for loc_elmt in item_elmt:
                                    for point_elmt in loc_elmt:
                                        if point_elmt.tag == "begin":
                                            begin = int(point_elmt.text)
                                        elif point_elmt.tag == "end":
                                            end = int(point_elmt.text)
                                    locations.append((begin, end))
                        annotated_terms[word] = locations
                        contain_terms_elmt = True
                elif document_elmt.tag in composites:
                    new_document_attrs[document_elmt.tag] = [item.text for item in document_elmt]
                else:
                    new_document_attrs[document_elmt.tag] = document_elmt.text
            if self.annotations and new_document_attrs["document_id"] in self.annotations:  # annotation file
                new_document_attrs["terms"] = self.annotations[new_document_attrs["document_id"]]
                self.add_document(**new_document_attrs)
                self.has_terms_locations = True  # at least 1 with terms
            elif contain_terms_elmt:  # there is no annotation file but terms element exist
                new_document_attrs["terms"] = annotated_terms
                self.add_document(**new_document_attrs)
                self.has_terms_locations = True
            elif self.annotations is None:  # there is no annotation file and no terms element
                self.add_document(**new_document_attrs)

    def read_from_folder(self, root_folder):
        in_folders = [folder for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))]
        for in_folder in in_folders:
            xml_files = [f for f in os.listdir(os.path.join(root_folder, in_folder)) if f.endswith(".xml")]
            for xml_file in xml_files:
                self.read_from_xml(os.path.join(root_folder, in_folder, xml_file))

    def get_document_ids(self):
        return [document.document_id for document in self.iter_documents()]

    def get_sample(self, n, excluded_ids=None):
        sample_corpus = Corpus()
        indices = list(range(len(self)))
        random.shuffle(indices)
        acquired_count = 0
        i = 0
        while acquired_count < n and i < len(indices):
            document = self[indices[i]]
            i += 1
            document_id = document.document_id.text
            if excluded_ids and document_id in excluded_ids: continue
            sample_corpus.add_document_from_element(document)
            acquired_count += 1
        return sample_corpus

    def get_more_sample(self, n, json1_filename):
        existing_ids = []
        with open(json1_filename, "r") as json1_file:
            lines = json1_file.readlines()
        for line in lines:
            json_news = json.loads(line)
            current_id = md5(json_news["text"].split("|")[0].encode("utf-8")).hexdigest()[-6:]
            existing_ids.append(current_id)
        return self.get_sample(n, existing_ids)

    def get_documents_by_ids(self, ids):
        subset_corpus = Corpus()
        for document in self:
            if document.document_id in ids:
                subset_corpus.add_document_from_element(document)
        return subset_corpus

    def get_documents_by_urls(self, urls):
        subset_corpus = Corpus()
        for document in self:
            if document.url.text in urls:
                subset_corpus.add_document_from_element(document)
        return subset_corpus

    def get_annotated_terms_as_csv(self, csv_path):
        with open(csv_path, "w") as csv_file:
            fieldnames = ["document_id", "terms"]
            csv_writer = DictWriter(csv_file, fieldnames)
            csv_writer.writeheader()
            for doc in self.iter_documents():
                document_id = doc.document_id.text
                all_terms = [term.word.text.lower() for term in doc.terms.term]
                csv_writer.writerow({"document_id": document_id, "terms": "|".join(all_terms)})
        return True

    def train_test_split(self, test_size, random_seed=1337):
        dev_c = Corpus()
        test_c = Corpus()
        n = len(self) * test_size
        indices = list(range(len(self)))
        random.seed(random_seed)
        random.shuffle(indices)
        i = 0
        while i < len(indices):
            document = self[indices[i]]
            if i < n:
                dev_c.add_document_from_element(document)
            else:
                test_c.add_document_from_element(document)
            i += 1
        return dev_c, test_c

    def write_to_core_nlp_xmls(self, output_folder):
        # term_locs = []
        # term_state = ["O", "B", "I"]
        # stanza_nlp_w_ssplit = stanza.Pipeline(
        #     "en",
        #     processors={"tokenize": "gum", "ner": "default", "lemma": "gum", "pos": "gum", "depparse": "gum"},
        #     verbose=False
        # )
        # stanza_nlp_no_ssplit = stanza.Pipeline(
        #     "en",
        #     processors={"tokenize": "gum", "ner": "default", "lemma": "gum", "pos": "gum", "depparse": "gum"},
        #     verbose=False,
        #     tokenize_no_ssplit=True
        # )
        stanza_nlp = stanza.Pipeline(
            "en",
            processors={"tokenize": "gum", "ner": "default", "lemma": "gum", "pos": "gum", "depparse": "gum"},
            verbose=False,
            tokenize_no_ssplit=True
        )

        # assumed sentence is split
        def annotate_sentence(sentence):
            # annotated_text = stanza_nlp_no_ssplit(sentences) if no_ssplit else stanza_nlp_w_ssplit(sentences)
            annotated_text = stanza_nlp(sentence)
            annotated_sentences = []
            head_dict = {0: "root"}
            for sentence in annotated_text.sentences:
                annotated_sentence = []
                for token in sentence.tokens:
                    if len(token.words) > 1: log.info(token)
                    else:
                        word = token.words[0]
                        misc = dict(token_misc.split("=") for token_misc in word.misc.split("|"))
                        word_id = int(word.id)
                        head_dict[word_id] = word.text
                        start_char = buffer_offset + int(misc["start_char"])
                        end_char = buffer_offset + int(misc["end_char"])
                        annotated_sentence.append({
                            "id": word_id,
                            "word": word.text,
                            "pos": word.xpos,
                            "lemma": word.lemma,
                            "deprel": word.deprel,
                            "deprel_head_id": word.head,
                            "character_offset_begin": start_char,
                            "character_offset_end": end_char,
                            "ner": token.ner
                            # "term_tag": term_state[bisect(term_locs, start_char) % 3] if len(term_locs) > 0 else None
                        })
                for token in annotated_sentence:
                    token["deprel_head_text"] = head_dict[token["deprel_head_id"]]
                annotated_sentences.append(annotated_sentence)
            return annotated_sentences

        for document in tqdm(self.iter_documents(), total=len(self)):
            document_id = document.document_id.text
            if f"{document_id}.xml" not in os.listdir(output_folder):
                buffer_offset = 0
                title = document.title.text
                term_locs = []
                if self.has_terms_locations:
                    for term in document.terms.term:
                        for location in term.locations.location:
                            insort(term_locs, int(location.begin.text)-0.5)
                            insort(term_locs, int(location.begin.text)+0.5)
                            insort(term_locs, int(location.end.text))
                annotated_title = annotate_sentence(title)
                buffer_offset += len(title) + 1
                annotated_content = []
                for p in document.content.p:
                    if len(p.text.strip()) > 0:
                        text = p.text.strip()
                        p_sents = nltk.tokenize.sent_tokenize(text)
                        for p_sent in p_sents:
                            annotated_content += annotate_sentence(p_sent)
                            buffer_offset += len(p_sent) + 1
                core_nlp_document = StanfordCoreNLPDocument()
                core_nlp_document.from_sentences(annotated_title, annotated_content)
                core_nlp_document.write_xml_to(os.path.join(output_folder, f"{document_id}.xml"))

    def write_to_jsonl(self, jsonl_path):
        # terms_found = False
        with open(jsonl_path, "w") as out_file:
            for document in self.iter_documents():
                # if document.terms.countchildren() > 0:
                #     labels = []
                #     for term in document.terms.term:
                #         for location in term.locations.location:
                #             labels.append([int(location.begin.text), int(location.end.text), "UNK"])
                doc_id = document.document_id.text
                text = {
                    "text": "|".join(
                        [document.title.text] + [p.text for p in document.content.p]
                    ),
                    "meta": {
                        "doc_id": doc_id
                    }
                }
                json.dump(html.unescape(text), out_file)
                out_file.write("\n")
                # terms_found = True
        # if not terms_found:
        #     raise ValueError("No terms found. Provide XML with terms to create a JSONL file.")


class StanfordCoreNLPCorpus(XMLBase):

    def __init__(self, core_nlp_folder):
        super().__init__("root", "document")
        self.root = objectify.Element("root")
        self.read_from_folder(core_nlp_folder)

    def read_from_folder(self, core_nlp_folder):
        filenames = [filename for filename in os.listdir(core_nlp_folder) if filename.endswith(".xml")]
        for filename in filenames:
            doc = StanfordCoreNLPDocument()
            doc.read_from_xml(os.path.join(core_nlp_folder, filename))
            doc = doc.root.document
            doc.set("id", filename.split(".")[0])
            self.root.append(doc)

    def get_summary(self):
        doc_stats = {}
        vocab_stats = {
            "vocabs": defaultdict(int),
            "verbs": defaultdict(int),
            "nouns": defaultdict(int),
            "adjs": defaultdict(int),
            "ORG": defaultdict(int),
            "DATE": defaultdict(int),
            "PERSON": defaultdict(int),
            "GPE": defaultdict(int),
            "CARDINAL": defaultdict(int),
            "FAC": defaultdict(int)
        }
        for doc in self.iter_documents():
            num_sentences = doc.sentences.countchildren()
            num_sentences_w_ne = 0
            num_tokens = 0
            num_nouns = 0
            num_verbs = 0
            num_adjs = 0
            num_ner = 0
            unique_lemma = []
            ners = {}
            for sentence in doc.sentences.sentence:
                contain_ne = False
                for token in sentence.tokens.token:
                    num_tokens += 1
                    lemma = token.lemma.text
                    if lemma not in unique_lemma:
                        unique_lemma.append(lemma)
                    vocab_stats["vocabs"][lemma] += 1
                    if token.POS.text[0] == "N":
                        num_nouns += 1
                        vocab_stats["nouns"][lemma] += 1
                    if token.POS.text[0] == "V":
                        num_verbs += 1
                        vocab_stats["verbs"][lemma] += 1
                    if token.POS.text[0] == "J":
                        num_adjs += 1
                        vocab_stats["adjs"][lemma] += 1
                    if token.ner.text != "O":
                        num_ner += 1
                        if not contain_ne:
                            num_sentences_w_ne += 1
                            contain_ne = True
                        ner_type = token.ner.text.split("-")[1]
                        if ner_type not in ners:
                            ners[ner_type] = 1
                        else:
                            ners[ner_type] += 1
                        if ner_type in vocab_stats:
                            word = token.word.text
                            vocab_stats[ner_type][word] += 1
            doc_stats[doc.get("id")] = {
                "#sents": num_sentences,
                "#sents_w_ne": num_sentences_w_ne,
                "#toks": num_tokens,
                "#nouns": num_nouns,
                "#verbs": num_verbs,
                "#adjs": num_adjs,
                "#ner": num_ner,
                "unique_lemma": len(unique_lemma)
            }
            for ne in ners:
                doc_stats[doc.get("id")][f"#ne_{ne}"] = ners[ne]
        return doc_stats, vocab_stats


class StanfordCoreNLPDocument(XMLBase):

    def __init__(self):
        super().__init__("root", "document")
        self.root = objectify.Element("root")
        self.sentence_id = 1

    def read_from_xml(self, xml_file):
        with open(xml_file) as f:
            xml = f.read()
        self.root = objectify.fromstring(xml)

    def from_sentences(self, title_sentences, content_sentences):
        document = objectify.Element("document")
        document.sentences = objectify.Element("sentences")
        sentences_element = self.process_sentences(title_sentences, "title", "title") + \
            self.process_sentences(content_sentences, "content", "bodyText")
        document.sentences.sentence = sentences_element
        self.root.append(document)

    def process_sentences(self, sentences, section_name, section_type):

        def process_sentence(sent):
            tokens = objectify.Element("tokens")
            tokens_element = []
            for token in sent:
                token_id = token["id"]
                token_element = objectify.Element("token", id=str(token_id))
                token_element.word = token["word"]
                token_element.lemma = token["lemma"]
                token_element.CharacterOffsetBegin = token["character_offset_begin"]
                token_element.CharacterOffsetEnd = token["character_offset_end"]
                token_element.POS = token["pos"]
                token_element.deprel = token["deprel"]
                token_element.deprel_head_id = token["deprel_head_id"]
                token_element.deprel_head_text = token["deprel_head_text"]
                # token_element.term_tag = token["term_tag"]
                token_element.ner = token["ner"]
                tokens_element.append(token_element)
            tokens.token = tokens_element
            return tokens

        sentences_element = []
        for sentence in sentences:
            sentence_element = objectify.Element(
                "sentence",
                section=section_name,
                type=section_type,
                id=str(self.sentence_id)
            )
            sentence_element.tokens = process_sentence(sentence)
            sentences_element.append(sentence_element)
            self.sentence_id += 1
        return sentences_element

    def find_sentences_w_term(self, term):
        sentences = []
        term_tokens = term.split()
        for sentence in self.root.document.sentences.sentence:
            token_words = [token.word.text.lower() for token in sentence.tokens.token]
            for i in range(len(token_words)):
                if token_words[i:i+len(term_tokens)] == term_tokens:
                    sentences.append(sentence)
        return sentences


def doc_stats():
    # DOCUMENT STATISTICS
    import string
    xml_folders = [
        ("Train Set", "../data/processed/news/relevant/train/"),
        ("Dev Set", "../data/processed/news/relevant/dev/"),
        ("Test Set", "../data/processed/news/relevant/test/"),
        ("Online Docs", "../data/processed/online_docs/snlp/")
    ]
    corpus_stats = {}
    for xml_name, xml_folder in xml_folders:
        corpus_stats[xml_name] = {}
        corpus = StanfordCoreNLPCorpus(xml_folder)
        corpus_stats[xml_name] = {
            "length": len(corpus)
        }
        corpus_summary, vocab_summary = corpus.get_summary()
        for doc_id in corpus_summary:
            for key in corpus_summary[doc_id]:
                if key not in corpus_stats[xml_name]:
                    corpus_stats[xml_name][key] = corpus_summary[doc_id][key]
                else:
                    corpus_stats[xml_name][key] += corpus_summary[doc_id][key]
        # corpus_stats[xml_folder]["totals"] = summary_total
        # corpus_stats[xml_folder]["tokens"] = {}
        # for key in vocab_summary:
        #     sorted_vocab = [
        #         (k, v) for k, v in sorted(vocab_summary[key].items(), reverse=True, key=lambda item: item[1])
        #         if k not in list(STOP_WORDS) and k not in string.punctuation
        #     ]
        #     corpus_stats[xml_folder]["tokens"][key] = sorted_vocab[:20]
    # TERMS STATISTICS
    terms_csvs = [
        ("Dev Set", "../data/processed/news/relevant/dev_terms.csv"),
        ("Test Set", "../data/processed/news/relevant/test_terms.csv"),
        ("Online Docs", "../data/processed/online_docs/online_docs_terms.csv")
    ]
    terms_stats = {}
    for csv_name, csv_filepath in terms_csvs:
        stats = {
            "terms_p_document": [],
            "words_p_terms": [],
        }
        with open(csv_filepath, "r") as f:
            reader = DictReader(f)
            for row in reader:
                terms = row["terms"].split("|")
                stats["terms_p_document"].append(len(terms))
                for term in terms:
                    stats["words_p_terms"].append(len([t for t in term.split(" ") if len(t) > 0]))
        terms_stats[csv_name] = stats
    with open("../results/stats/stats-table.ltx", "w") as f:
        # header
        f.write(" & ".join([" "] + ["\\textbf{" + c + "}" for c in corpus_stats]) + "\\\\ \\hline\n")
        f.write(" & ".join(["Total documents"] + [str(corpus_stats[c]["length"]) for c in corpus_stats]) + "\\\\\n")
        f.write(" & ".join(["Total sentences"] + [str(corpus_stats[c]["#sents"]) for c in corpus_stats]) + "\\\\\n")
        f.write(" & ".join(
            ["Total sentences w/NE"]
            + [str(corpus_stats[c]["#sents_w_ne"]) for c in corpus_stats]) + "\\\\\n"
        )
        f.write(" & ".join(["Total tokens"] + [str(corpus_stats[c]["#toks"]) for c in corpus_stats]) + "\\\\\n")
        f.write(" & ".join(["Total nouns"] + [str(corpus_stats[c]["#nouns"]) for c in corpus_stats]) + "\\\\\n")
        f.write(" & ".join(["Total verbs"] + [str(corpus_stats[c]["#verbs"]) for c in corpus_stats]) + "\\\\\n")
        f.write(" & ".join(["Total adjectives"] + [str(corpus_stats[c]["#adjs"]) for c in corpus_stats]) + "\\\\\n")
        f.write(
            " & ".join(
                ["Total terms", "-"] + [str(sum(terms_stats[c]["terms_p_document"])) for c in terms_stats]
            ) + "\\\\\n"
        )
        f.write(" & ".join(["Unique Lemma"] + [str(corpus_stats[c]["unique_lemma"]) for c in corpus_stats]) + "\\\\\n")
        f.write(" & ".join(
            ["Unique Lemma Ratio"]
            + ["{:.2f}".format(corpus_stats[c]["unique_lemma"] / corpus_stats[c]["#toks"]) for c in corpus_stats]
        ) + "\\\\ \\hline\n")
        f.write(" & ".join(
            ["Sentences per document"]
            + ["{:.2f}".format(corpus_stats[c]["#sents"] / corpus_stats[c]["length"]) for c in corpus_stats]
        ) + "\\\\\n")
        f.write(
            " & ".join(
                ["Terms per document", "-"] + ["{:.2f}".format(
                    sum(terms_stats[c]["terms_p_document"]) / len(terms_stats[c]["terms_p_document"])
                ) for c in terms_stats]
            ) + "\\\\\n"
        )
        f.write(" & ".join(
            ["Tokens per sentence"]
            + ["{:.2f}".format(corpus_stats[c]["#toks"] / corpus_stats[c]["#sents"]) for c in corpus_stats]
        ) + "\\\\\n")
        f.write(" & ".join(
            ["Nouns per sentence"]
            + ["{:.2f}".format(corpus_stats[c]["#nouns"] / corpus_stats[c]["#sents"]) for c in corpus_stats]
        ) + "\\\\\n")
        f.write(" & ".join(
            ["Verbs per sentence"]
            + ["{:.2f}".format(corpus_stats[c]["#verbs"] / corpus_stats[c]["#sents"]) for c in corpus_stats]
        ) + "\\\\\n")
        f.write(" & ".join(
            ["Adjectives per sentence"]
            + ["{:.2f}".format(corpus_stats[c]["#adjs"] / corpus_stats[c]["#sents"]) for c in corpus_stats]
        ) + "\\\\\n")
        f.write(
            " & ".join(
                ["Tokens per terms", "-"] + ["{:.2f}".format(
                    sum(terms_stats[c]["words_p_terms"]) / len(terms_stats[c]["words_p_terms"])
                ) for c in terms_stats]
            ) + "\\\\\n"
        )

    with open("../results/stats/ner-table.ltx", "w") as f:
        f.write(" & ".join([" "] + ["\\textbf{" + c + "}" for c in corpus_stats]) + "\\\\ \\hline\n")
        sum_set = {c: 0 for c in corpus_stats}
        for ne in ["ORG", "DATE", "PERSON", "GPE", "CARDINAL", "FAC"]:
            f.write(
                " & ".join(
                    [ne] + ["{:.1f}\\%".format(
                        corpus_stats[c][f"#ne_{ne}"] * 100 / corpus_stats[c]["#ner"]
                    ) for c in corpus_stats]
                ) + "\\\\\n"
            )
            for c in corpus_stats:
                sum_set[c] += corpus_stats[c][f"#ne_{ne}"]
        f.write(
            " & ".join(
                ["Others"]
                + ["{:.1f}\\%".format(
                    (corpus_stats[c]["#ner"] - sum_set[c]) * 100 / corpus_stats[c]["#ner"]
                ) for c in corpus_stats]
            ) + "\\\\\n"
        )


def train_test_split():
    # TRAIN/DEV/TEST SPLIT
    corpus_all = Corpus("../data/interim/all-v2.xml")
    anno_json = AnnotationJSON("../data/manual/labels/annotation.json1")
    # irrelevant
    log.info("Start processing irrelevant corpus")
    irrel_corpus = corpus_all.get_documents_by_ids(anno_json.irrelevants)
    irrel_train_corpus, irrel_dev_test_corpus = irrel_corpus.train_test_split(test_size=0.5)
    irrel_dev_corpus, irrel_test_corpus = irrel_dev_test_corpus.train_test_split(test_size=0.5)
    irrel_train_corpus.write_xml_to("../data/processed/irrelevant/train.xml")
    irrel_train_corpus.write_to_core_nlp_xmls("../data/processed/irrelevant/train/")
    irrel_dev_corpus.write_xml_to("../data/processed/irrelevant/dev.xml")
    irrel_dev_corpus.write_to_core_nlp_xmls("../data/processed/irrelevant/dev/")
    irrel_test_corpus.write_xml_to("../data/processed/irrelevant/test.xml")
    irrel_test_corpus.write_to_core_nlp_xmls("../data/processed/irrelevant/test/")
    # relevant
    log.info("Start processing relevant corpus")
    corpus_anno = Corpus("../data/interim/all-v2.xml", annotations=anno_json)
    corpus_ref = Corpus("../data/interim/lda_sampling_10p.xml")
    # training set
    excluded_ids = anno_json.irrelevants + corpus_anno.get_document_ids()
    rel_train_corpus = corpus_ref.get_sample(len(corpus_ref), excluded_ids=excluded_ids)
    rel_train_corpus.write_xml_to("../data/processed/relevant/train.xml")
    rel_train_corpus.write_to_core_nlp_xmls("../data/processed/relevant/train/")
    # dev/test set
    rel_dev_corpus, rel_test_corpus = corpus_anno.train_test_split(test_size=0.5)
    rel_dev_corpus.write_xml_to("../data/processed/relevant/dev.xml")
    rel_dev_corpus.write_to_core_nlp_xmls("../data/processed/relevant/dev")
    rel_test_corpus.write_xml_to("../data/processed/relevant/test.xml")
    rel_test_corpus.write_to_core_nlp_xmls("../data/processed/relevant/test")
    rel_test_corpus = Corpus("../data/processed/relevant/test.xml")
    rel_test_corpus.write_to_core_nlp_xmls("../data/processed/relevant/test")


def get_term_csv():
    # Get dev/test terms in CSV
    dev_rel_corpus = Corpus("../data/processed/news/relevant/dev.xml")
    test_rel_corpus = Corpus("../data/processed/news/relevant/test.xml")
    oldoc_corpus = Corpus("../data/processed/online_docs/online_docs.xml")
    dev_rel_corpus.get_annotated_terms_as_csv("../data/processed/news/relevant/dev_terms.csv")
    test_rel_corpus.get_annotated_terms_as_csv("../data/processed/news/relevant/test_terms.csv")
    oldoc_corpus.get_annotated_terms_as_csv("../data/processed/online_docs/online_docs_terms.csv")


def get_more_samples():
    corpus = StanfordCoreNLPCorpus("../data/test/core_nlp_samples")
    # corpus.write_xml_to("../data/interim/delete_me1.xml")
    # more_sample = corpus.get_more_sample(50, "../data/manual/backup-20200511.json1")
    # more_sample.write_to_jsonl("../data/processed/more_sample_50_2_lda15p.json1")


def process_html_excerpt():
    # corpus = Corpus("../data/interim/online_docs.xml")
    # corpus.write_xml_to("../data/interim/online_docs.2.xml")
    # corpus.write_to_jsonl("../data/manual/online_docs.json")
    anno_json = AnnotationJSON("../data/manual/online_docs.json1")
    corpus = Corpus(
        "../data/processed/online_docs/online_docs.xml",
        annotations=anno_json
    )
    corpus.write_xml_to("../data/interim/online_docs_annotated.xml")
    # corpus = Corpus("../data/interim/online_docs_annotated.xml")
    # corpus.write_to_core_nlp_xmls("../data/processed/online_docs/")


def check_duplicate_dev_test():
    dev_list = set([filename for filename in os.listdir("../data/processed/news/relevant/dev/")])
    test_list = set([filename for filename in os.listdir("../data/processed/news/relevant/test/")])
    print(f"DEV unique ids: {len(dev_list)}")
    print(f"TEST unique ids: {len(test_list)}")
    print(f"Duplicate: {dev_list.intersection(test_list)}")


if __name__ == "__main__":
    doc_stats()
