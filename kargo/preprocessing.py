import os
import nltk
from kargo.corpus import Corpus, StanfordCoreNLPDocument
from kargo import logger
log = logger.get_logger(__name__, logger.INFO)


def combine_xmls(root_folder):
    in_folders = [folder for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))]
    corpus = Corpus()
    for in_folder in in_folders:
        xml_files = [f for f in os.listdir(os.path.join(root_folder, in_folder)) if f.endswith(".xml")]
        for xml_file in xml_files:
            corpus.read_from_xml(os.path.join(root_folder, in_folder, xml_file))
    return corpus


def filter_empty(corpus):
    new_corpus = Corpus()
    for document in corpus.iter_documents():
        if document.content.countchildren() > 0:
            new_corpus.add_document_from_element(document)
    return new_corpus


def write_core_nlp_xmls(corpus, output_folder, host="localhost", port=9000):

    def annotate_sentence(sentences):
        properties = {"annotators": "tokenize,ssplit,pos,lemma"}
        annotated_text = parser.api_call(sentences, properties=properties)
        annotated_sentences = []
        for sentence in annotated_text["sentences"]:
            annotated_sentence = []
            for token in sentence["tokens"]:
                annotated_sentence.append({
                    "word": token["word"],
                    "pos": token["pos"],
                    "lemma": token["lemma"],
                    "character_offset_begin": token["characterOffsetBegin"],
                    "character_offset_end": token["characterOffsetEnd"]
                })
            annotated_sentences.append(annotated_sentence)
        return annotated_sentences

    parser = nltk.CoreNLPParser(url=f"http://{host}:{port}")
    for document in corpus.iter_documents():
        document_id = document.document_id.text
        title = document.title.text
        annotated_title = annotate_sentence(title)
        annotated_content = []
        for p in document.content.p:
            annotated_content += annotate_sentence(p.text)
        core_nlp_corpus = StanfordCoreNLPDocument(annotated_title, annotated_content)
        core_nlp_corpus.write_xml_to(os.path.join(output_folder, f"{document_id}.xml"))


if __name__ == "__main__":
    combined_corpus = combine_xmls("../data/scraped/")
    filtered_corpus = filter_empty(combined_corpus)
    filtered_corpus.write_xml_to("../data/interim/all.xml")
    manual_corpus = Corpus("../data/processed/random_sample_annotated.xml")
    write_core_nlp_xmls(manual_corpus, "../data/processed/stanford_core_nlp_xmls/")
