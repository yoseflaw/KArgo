from kargo.corpus import StanfordCoreNLPCorpus

if __name__ == "__main__":
    # check no duplicate
    existing_ids = {}
    xml_folders = [
        "data/processed/irrelevant/train",
        "data/processed/irrelevant/dev",
        "data/processed/irrelevant/test",
        "data/processed/relevant/train",
        "data/processed/relevant/dev",
        "data/processed/relevant/test",
    ]
    # for xml_folder in xml_folders:
    #     corpus = StanfordCoreNLPCorpus(xml_folder)
    #     for doc in corpus.iter_documents():
    #         doc_id = doc.get("id")
    #         if doc_id in existing_ids:
    #             print(f"ID exist: {doc_id} in {xml_folder} and {existing_ids[doc_id]}")
    #         else:
    #             existing_ids[doc_id] = xml_folder
    # number of documents
    docs = {}
    for xml_folder in xml_folders:
        corpus = StanfordCoreNLPCorpus(xml_folder)
        print(xml_folder)
        print("number of documents:", len(corpus))
        for doc in corpus.iter_documents():
            print("number of sentences: ", doc.sentences.countchildren())
    # number of sentences

    # average sentences per document

    # number of words

    # average words per sentence

    # count of NE

    # count of POS

    # most used Verb

    # most used Noun

    # Occurrence of terms in document (position)