from hashlib import md5
import os
import random
import json
import html
from bisect import bisect, insort
from tqdm import tqdm
from lxml.objectify import Element
from lxml import etree, objectify
import stanza
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


class Corpus(XMLBase):
    annotation_regex = r"\[\[(.+?)\]\]"

    def __init__(self, xml_input=None, annotation_file=None):
        super().__init__("corpus", "document")
        self.corpus = Element("corpus")
        self.url_indices = []
        self.has_terms_locations = False
        self.annotations = self.process_json_annotation(annotation_file) if annotation_file else None
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
            [term.text for term in document_elmt.terms.term] if document_elmt.terms.countchildren() > 0 else None,
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

    def get_documents_by_urls(self, urls):
        subset_corpus = Corpus()
        for document in self:
            if document.url.text in urls:
                subset_corpus.add_document_from_element(document)
        return subset_corpus

    def process_json_annotation(self, annotation_file):
        with open(annotation_file, "r") as f_anno:
            annotations = f_anno.readlines()
        annotations_dict = {}
        for annotation in annotations:
            doc = json.loads(annotation)
            title = doc["text"].split("|")[0]
            text = doc["text"]
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
            if not has_irrelevant:
                annotations_dict[doc_id] = annotation_mapping
        return annotations_dict

    def write_to_core_nlp_xmls(self, output_folder):
        term_locs = []
        term_state = ["O", "B", "I"]
        stanza_nlp_w_ssplit = stanza.Pipeline(
            "en",
            processors={"tokenize": "lines", "ner": "default", "lemma": "lines", "pos": "gum", "depparse": "lines"},
            verbose=False
        )
        stanza_nlp_no_ssplit = stanza.Pipeline(
            "en",
            processors={"tokenize": "lines", "ner": "default", "lemma": "lines", "pos": "gum", "depparse": "lines"},
            verbose=False,
            tokenize_no_ssplit=True
        )

        def annotate_sentence(sentences, no_ssplit):
            annotated_text = stanza_nlp_no_ssplit(sentences) if no_ssplit else stanza_nlp_w_ssplit(sentences)
            annotated_sentences = []
            head_dict = {0: "root"}
            for sentence in annotated_text.sentences:
                annotated_sentence = []
                for token in sentence.tokens:
                    if len(token.words) > 1: print(token)
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
                annotated_title = annotate_sentence(title, no_ssplit=True)
                buffer_offset += len(title) + 1
                annotated_content = []
                for p in document.content.p:
                    if len(p.text.strip()) > 0:
                        annotated_content += annotate_sentence(p.text, no_ssplit=False)
                        buffer_offset += len(p.text) + 1
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
                text = {
                    "text": "|".join(
                        [document.title.text] + [p.text for p in document.content.p]
                    ),
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


if __name__ == "__main__":
    corpus = Corpus("../data/interim/lda_sampling_15p.xml")
    # corpus.write_to_core_nlp_xmls("../data/processed/scnlp_lda_all")
    # n_sample = 10
    # sampled_corpus = corpus.get_sample(n_sample)
    # sampled_corpus.write_xml_to("../data/test/samples_news_clean_random.xml")
    # clean_corpus = Corpus(
    #     "../data/processed/lda_sampling_15p.xml",
    #     annotation_file="../data/manual/backup-20200426.json1"
    # )
    # clean_corpus.write_xml_to("../data/processed/lda_sampling_15p.annotated.xml")
    # clean_corpus.write_to_core_nlp_xmls("../data/processed/scnlp_xmls/")
    # corpus = StanfordCoreNLPCorpus("../data/test/core_nlp_samples")
    # corpus.write_xml_to("../data/interim/delete_me1.xml")
    more_sample = corpus.get_more_sample(50, "../data/manual/backup-20200511.json1")
    more_sample.write_to_jsonl("../data/processed/more_sample_50_2_lda15p.json1")
