import os
from kargo.corpus import Corpus
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


if __name__ == "__main__":
    combined_corpus = combine_xmls("../data/scraped/")
    filtered_corpus = filter_empty(combined_corpus)
    filtered_corpus.write_xml_to("../data/interim/all.xml")
