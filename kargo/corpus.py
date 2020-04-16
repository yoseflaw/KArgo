from hashlib import md5
import re
import os
import random
import json
import html
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
        return len(getattr(self.get_root(), self.document_tag))

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

    def __init__(self, xml_input=None, is_annotated=False):
        super().__init__("corpus", "document")
        self.corpus = Element("corpus")
        self.url_indices = []
        if xml_input:
            if xml_input and not os.path.exists(xml_input):
                raise FileNotFoundError(f"{xml_input} not found. Check the path again.")
            elif os.path.isfile(xml_input):
                self.is_annotated = is_annotated
                self.read_from_xml(xml_input)
            else:
                self.is_annotated = is_annotated
                self.read_from_folder(xml_input)
        else:
            self.is_annotated = False

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
        new_document.document_id = md5(url.encode("utf-8")).hexdigest()[-6:] if document_id is None or \
            len(document_id) == 0 else document_id
        new_document.url = url
        new_document.title = Corpus.unicodify(title)
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
        has_terms = False
        for document in corpus_root:
            new_document_attrs = {}
            unique_terms = {}
            buffer_len = 0
            for document_elmt in document:
                if document_elmt.tag == "category":
                    new_document_attrs["categories"] = document_elmt.text.split(";") if document_elmt.text else []
                elif document_elmt.tag == "title" and self.is_annotated:
                    new_title = self.process_annotation(Corpus.unicodify(document_elmt.text), unique_terms, buffer_len)
                    new_document_attrs["title"] = new_title
                    buffer_len += len(new_title) + 1
                elif document_elmt.tag == "content" and self.is_annotated:
                    new_ps = []
                    for item in document_elmt:
                        new_p = self.process_annotation(Corpus.unicodify(item.text), unique_terms, buffer_len)
                        new_ps.append(new_p)
                        buffer_len += len(new_p) + 1
                    new_document_attrs["content"] = new_ps
                elif document_elmt.tag == "terms":
                    terms = {}
                    for term_elmt in document_elmt:
                        locations = []
                        word = None
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
                        terms[word] = locations
                    has_terms = True
                    new_document_attrs["terms"] = terms
                elif document_elmt.tag in composites:
                    new_document_attrs[document_elmt.tag] = [item.text for item in document_elmt]
                else:
                    new_document_attrs[document_elmt.tag] = document_elmt.text
            if not has_terms and self.is_annotated:
                new_document_attrs["terms"] = unique_terms
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

    def get_documents_by_urls(self, urls):
        subset_corpus = Corpus()
        for document in self:
            if document.url.text in urls:
                subset_corpus.add_document_from_element(document)
        return subset_corpus

    def process_annotation(self, text, unique_terms, buffer_len):
        terms = re.findall(self.annotation_regex, text)
        new_text = re.sub(self.annotation_regex, r"\1", text)
        for term in terms:
            locations = [
                (buffer_len + m.start(), buffer_len + m.end()) for m in re.finditer(re.escape(term), new_text)
            ]
            if term in unique_terms:
                unique_terms[term].update(locations)
            else:
                unique_terms[term] = set(locations)
        return new_text

    def write_to_core_nlp_xmls(self, output_folder):

        def annotate_sentence(sentences, nlp):
            annotated_text = nlp(sentences)
            annotated_sentences = []
            head_dict = {0: "root"}
            for sentence in annotated_text.sentences:
                annotated_sentence = []
                for token in sentence.words:
                    misc = dict(token_misc.split("=") for token_misc in token.misc.split("|"))
                    token_id = int(token.id)
                    head_dict[token_id] = token.text
                    annotated_sentence.append({
                        "id": token_id,
                        "word": token.text,
                        "pos": token.xpos,
                        "lemma": token.lemma,
                        "deprel": token.deprel,
                        "deprel_head_id": token.head,
                        "character_offset_begin": misc["start_char"],
                        "character_offset_end": misc["end_char"]
                    })
                for token in annotated_sentence:
                    token["deprel_head_text"] = head_dict[token["deprel_head_id"]]
                annotated_sentences.append(annotated_sentence)
            return annotated_sentences

        stanza_nlp = stanza.Pipeline("en", package="gum", processors="tokenize,pos,lemma,depparse", verbose=False)
        for document in tqdm(self.iter_documents(), total=len(self)):
            document_id = document.document_id.text
            title = document.title.text
            annotated_title = annotate_sentence(title, stanza_nlp)
            annotated_content = []
            for p in document.content.p:
                annotated_content += annotate_sentence(p.text, stanza_nlp)
            core_nlp_corpus = StanfordCoreNLPDocument(annotated_title, annotated_content)
            core_nlp_corpus.write_xml_to(os.path.join(output_folder, f"{document_id}.xml"))

    def write_annotation_to_jsonl(self, jsonl_path):
        terms_found = False
        with open(jsonl_path, "w") as out_file:
            for document in self.iter_documents():
                if document.terms.countchildren() > 0:
                    labels = []
                    for term in document.terms.term:
                        for location in term.locations.location:
                            labels.append([int(location.begin.text), int(location.end.text), "UNK"])
                    text = {
                        "text": "|".join(
                            [html.unescape(document.title.text)]\
                            + [p.text for p in document.content.p]
                        ),
                        # "text": document.title.text,
                        # "labels": labels
                    }
                    json.dump(html.unescape(text), out_file)
                    out_file.write("\n")
                    terms_found = True
        if not terms_found:
            raise ValueError("No terms found. Provide XML with terms to create a JSONL file.")


class StanfordCoreNLPDocument(XMLBase):

    def __init__(self, title_sentences, content_sentences):
        super().__init__("root", "document")
        self.root = objectify.Element("root")
        document = objectify.Element("document")
        document.sentences = objectify.Element("sentences")
        self.sentence_id = 1
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
                token_element.depreal_head_id = token["deprel_head_id"]
                token_element.depreal_head_text = token["deprel_head_text"]
                tokens_element.append(token_element)
            tokens.token = tokens_element
            return tokens

        sentences_element = []
        for sentence in sentences:
            sentence_element = objectify.Element("sentence",
                                                 section=section_name, type=section_type, id=str(self.sentence_id))
            sentence_element.tokens = process_sentence(sentence)
            sentences_element.append(sentence_element)
            self.sentence_id += 1
        return sentences_element



if __name__ == "__main__":
    corpus = Corpus("../data/manual/random_sample_annotated.xml", is_annotated=True)
    corpus.write_xml_to("../data/processed/random_sample_annotated.xml")
    corpus.write_annotation_to_jsonl("../data/processed/random_sample_annotated.jsonl")
