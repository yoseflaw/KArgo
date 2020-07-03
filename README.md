# `KArgo` - Knowledge Acquistion from Special Cargo Text

`KArgo` is the automatic labeling implementation for my Master's Thesis:

**Automatic Knowledge Acquisition for the Special Cargo Services Domain with Unsupervised Entity and Relation Extraction**

The automatic labeling consists of two parts: entity extraction and relation clustering. The goal of the process is to produce labels automatically from the special cargo news articles with unsupervised approach. 

<img src="https://github.com/yoseflaw/KArgo/blob/master/images/kargo.png" alt="Kargo Scheme"/>

The repository contains the following folders:
* data
    * annotations
        * automatic/terms: the converted automatically extracted terms from the training set.
        * relations: manual annotations for the relation, as downloaded from [Doccano](https://github.com/doccano/doccano) platform.
        * terms: manual annotations for the term, as downloaded from [Doccano](https://github.com/doccano/doccano) platform.
    * interim: intermediate files for term/relation extraction.
        * cargo_df.tsv.gz: TF-IDF calculation of special cargo news articles for statistical keyphrase extraction.
        * to_anno.jsonl: preprocessed news text following [Doccano](https://github.com/doccano/doccano) jsonl input format.
    * processed: preprocessed files from the raw format
        * news: news articles as extracted from five different cargo news sites. Some news articles belong to the irrelevant category as marked by an expert during manual annotation.
        * online_docs: similar to news, but text taken from ten HTML/PDF files.
        * lda_sampling_15p.xml: the result of automatic document filtering with LDA. Taken from one most representative topic (from ten topics) with minimum probability 0.85.
    * scraped: samples of the five scraped news sites
    * test: files for testing purposes
* kargo: main source code folder
    * tests: unit tests (very basic stuff)
    * corpus.py: modules to process all the file formats. Contains the main corpus data structure `Corpus` and `StanfordCoreNLPCorpus`. XML data structure based on [lxml](https://lxml.de/).
    * evaluation.py: modules related to term extraction evaluation. For relation extraction, evaluation embedded in `relation.py`.
    * logger.py: a simple logging module.
    * relations.py: modules related to relation extraction, including the evaluation and document processing.
    * scraping.py: [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) based news pages scraping.
    * terms.py: modules related to term extraction, based on [pke](https://github.com/boudinfl/pke). Also includes a wrapper for [EmbedRank](https://github.com/swisscom/ai-research-keyphrase-extraction).
    * topic_modelling: document filtering using LDA.
* results:
    * extracted_relations: clustered relations as tested with various DBSCAN parameters (`eps` and `min_samples`). Files are named based on the parameter values.
        * labels: every co-occurrences with the corresponding relation label. `0` means no relation, `1` means otherwise.
        * relation_jsons: complete co-occurrences metadata, grouped by the cluster number from DBSCAN.
    * extracted_terms: unsupervised keyphrase extraction for each respective dataset.
    * plots: experiment results, produced using [Altair](https://altair-viz.github.io/).
    
Unsupervised Keyphrase Extraction results:

<img src="https://github.com/yoseflaw/KArgo/blob/master/results/plots/eval_20200630.png" alt="Keyphrase Extraction Results" />