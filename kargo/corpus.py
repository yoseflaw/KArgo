from hashlib import md5
import re
from lxml.objectify import Element
from lxml import etree, objectify


class XMLBase(object):

    def __init__(self, root_name):
        self.root_name = root_name

    def get_root(self):
        return getattr(self, self.root_name)

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
        super().__init__("corpus")
        self.corpus = Element("corpus")
        self.url_indices = []
        if xml_input:
            self.is_annotated = is_annotated
            self.read_from_xml(xml_input)

    def add_document(self, url, title, categories, published_time, content, author=None, topics=None, links=None,
                     terms=None, document_id=None):
        if url is None or len(url) == 0:
            raise KeyError("'url' is mandatory")
        elif url in self.url_indices:
            print(f"[LOG] Already exist URL={url}")
            return
        new_document = Element("document")
        new_document.document_id = md5(url.encode("utf-8")).hexdigest()[-6:] if document_id is None or \
            len(document_id) == 0 else document_id
        new_document.url = url
        new_document.title = title
        new_document.author = author
        new_document.published_time = published_time
        # handle lists
        new_document.terms = Element("terms")
        new_document.terms.term = terms
        new_document.categories = Element("categories")
        new_document.categories.category = categories
        new_document.topics = Element("topics")
        new_document.topics.topic = topics
        new_document.links = Element("links")
        new_document.links.link = links
        new_document.content = Element("content")
        new_document.content.p = content
        self.corpus.append(new_document)
        self.url_indices.append(url)

    def read_from_xml(self, input_path):
        composites = ["terms", "topics", "content", "links", "categories"]
        corpus_etree = etree.parse(input_path)
        corpus_root = corpus_etree.getroot()
        for document in corpus_root:
            new_document_attrs = {}
            unique_terms = set()
            for document_elmt in document:
                if document_elmt.tag == "category":
                    new_document_attrs["categories"] = document_elmt.text.split(";") if document_elmt.text else []
                elif document_elmt.tag == "title" and self.is_annotated:
                    terms = re.findall(r"\[\[(.+?)\]\]", document_elmt.text)
                    unique_terms.update(terms)
                    new_document_attrs["title"] = re.sub(self.annotation_regex, r"\1", document_elmt.text)
                elif document_elmt.tag == "content" and self.is_annotated:
                    for item in document_elmt:
                        terms = re.findall(r"\[\[(.+?)\]\]", item.text)
                        unique_terms.update(terms)
                    new_document_attrs["content"] = [re.sub(self.annotation_regex, r"\1", p.text) for p in document_elmt]
                elif document_elmt.tag in composites:
                    new_document_attrs[document_elmt.tag] = [item.text for item in document_elmt]
                else:
                    new_document_attrs[document_elmt.tag] = document_elmt.text
            if self.is_annotated:
                new_document_attrs["terms"] = list(unique_terms)
            self.add_document(**new_document_attrs)


class CoreNLPDocument(XMLBase):

    def __init__(self, title_sentences, content_sentences):
        super().__init__("root")
        self.root = objectify.Element("root")
        document = objectify.Element("document")
        document.sentences = objectify.Element("sentences")
        self.sentence_id = 1
        sentences_element = self.process_sentences(title_sentences, "title", "title") + \
            self.process_sentences(content_sentences, "content", "bodyText")
        document.sentences.sentence = sentences_element
        self.root.append(document)

    def process_sentences(self, sentences, section_name, section_type):
        sentences_element = []
        for sentence in sentences:
            sentence_element = objectify.Element("sentence",
                                                 section=section_name, type=section_type, id=str(self.sentence_id))
            sentence_element.tokens = self.process_sentence(sentence)
            sentences_element.append(sentence_element)
            self.sentence_id += 1
        return sentences_element

    def process_sentence(self, sentence):
        tokens = objectify.Element("tokens")
        tokens_element = []
        token_id = 1
        for token in sentence:
            token_element = objectify.Element("token", id=str(token_id))
            token_element.word = token["word"]
            token_element.lemma = token["lemma"]
            token_element.CharacterOffsetBegin = token["character_offset_begin"]
            token_element.CharacterOffsetEnd = token["character_offset_end"]
            token_element.POS = token["pos"]
            tokens_element.append(token_element)
            token_id += 1
        tokens.token = tokens_element
        return tokens


if __name__ == "__main__":
    corpus = Corpus("../data/test/samples_annotated.xml", is_annotated=True)
    in_xml = corpus.get_xml()
    print(in_xml)
