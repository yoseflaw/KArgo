import os
from datetime import date
from spacy.lang.en.stop_words import STOP_WORDS
from kargo import logger, corpus, scraping, extraction, evaluation
from pke.utils import compute_document_frequency, load_document_frequency_file
from pke.unsupervised import TfIdf, KPMiner, YAKE
from pke.unsupervised import SingleRank, TopicRank, PositionRank, MultipartiteRank
from pke.supervised import Kea
SCRAPED_DIR = "data/scraped"
INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"
MANUAL_DIR = "data/manual"
RESULTS_DIR = "results"
CORE_NLP_DIR = os.path.join(PROCESSED_DIR, "stanford_core_nlp_xmls")
EXTRACTED_DIR = os.path.join(RESULTS_DIR, "extracted_terms")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
log = logger.get_logger(__name__, logger.INFO)


def scraping_news_sites():
    log.info("Begin scraping processes")
    air_cargo_news_spider = scraping.AirCargoNewsSpider(
        seed_url="https://www.aircargonews.net/news-by-date/page/",
        output_folder=os.path.join(SCRAPED_DIR, "aircargonews.net")
    )
    log.info("Begin scraping aircargonews.net")
    air_cargo_news_spider.start(1, 2)
    air_cargo_week_spider = scraping.AirCargoWeekSpider(
        seed_url="https://www.aircargoweek.com/category/news-menu/page/",
        output_folder=os.path.join(SCRAPED_DIR, "aircargoweek.com")
    )
    log.info("Begin scraping aircargoweek.com")
    air_cargo_week_spider.start(1, 2)
    air_cargo_world_spider = scraping.AirCargoWorldSpider(
        seed_url="https://aircargoworld.com/category/news/page/",
        output_folder=os.path.join(SCRAPED_DIR, "aircargoworld.com")
    )
    log.info("Begin scraping aircargoworld.com")
    air_cargo_world_spider.start(1, 2)
    the_load_star_spider = scraping.TheLoadStarSpider(
        seed_url="https://theloadstar.com/category/news/page/",
        output_folder=os.path.join(SCRAPED_DIR, "theloadstar.com")
    )
    log.info("Begin scraping theloadstar.com")
    the_load_star_spider.start(1, 2)
    stat_times_spider = scraping.StatTimesSpider(
        seed_url="https://www.stattimes.com/category/air-cargo/page/",
        output_folder=os.path.join(SCRAPED_DIR, "stattimes.com")
    )
    log.info("Begin scraping stattimes.com")
    stat_times_spider.start(1, 2)


def combine_filter_sample_corpus():
    log.info(f"Begin combining from {SCRAPED_DIR}")
    combined_corpus = corpus.Corpus(SCRAPED_DIR)
    log.info("Begin filtering empty documents")
    combined_corpus.filter_empty()
    n_sample = 10
    log.info(f"Begin sampling, n={n_sample}")
    sampled_corpus = combined_corpus.get_sample(n_sample)
    log.info(f"Write sample.xml to {INTERIM_DIR}")
    sampled_corpus.write_xml_to(os.path.join(INTERIM_DIR, "sample.xml"))  # use dummy filename for now


def manual_term_annotation():
    log.info(f"Manual annotation assumed to be done with doccano, export results to {MANUAL_DIR}")
    # currently done manually to output of combine_filter_sample_corpus
    # assumed result in MANUAL_DIR
    pass


def process_manual_annotation():
    log.info(f"Begin incorporating manual annotation to the XML, result in {PROCESSED_DIR}")
    manual_corpus = corpus.Corpus(
        os.path.join(INTERIM_DIR, "random_sample_annotated.xml"),
        annotation_file=os.path.join(MANUAL_DIR, "annotation.json1")
    )
    manual_corpus.write_xml_to(os.path.join(PROCESSED_DIR, "random_sample_annotated.xml"))


def create_core_nlp_documents(core_nlp_folder):
    log.info(f"Begin preparing Core NLP Documents to {core_nlp_folder}")
    annotated_corpus = corpus.Corpus(os.path.join(PROCESSED_DIR, "random_sample_annotated.xml"))
    annotated_corpus.write_to_core_nlp_xmls(core_nlp_folder)
    compute_document_frequency(
        core_nlp_folder, os.path.join(INTERIM_DIR, "cargo_df.tsv.gz"),
        stoplist=list(STOP_WORDS)
    )


def extract_terms(core_nlp_folder):
    log.info("Begin Extraction")
    n = 15
    considered_pos = {"NOUN", "PROPN", "NUM", "ADJ", "ADP"}
    cargo_df = load_document_frequency_file(os.path.join(INTERIM_DIR, "cargo_df.tsv.gz"))
    # PKE: Tfidf
    log.info("Begin extraction with a PKE extractor: TF-IDF")
    tfidf_extractor = extraction.PKEBasedExtractor(TfIdf)
    tfidf_selection_params = {
        "stoplist": list(STOP_WORDS)
    }
    tfidf_weighting_params = {
        "df": cargo_df
    }
    tfidf_extractor.extract(
        core_nlp_folder, n,
        selection_params=tfidf_selection_params,
        weighting_params=tfidf_weighting_params,
        output_file=os.path.join(EXTRACTED_DIR, "tfidf.csv")
    )
    # PKE: KPM
    log.info("Begin extraction with a PKE extractor: KPMiner")
    kpm_extractor = extraction.PKEBasedExtractor(KPMiner)
    kpm_selection_params = {
        "lasf": 1,
        "cutoff": 200,
        "stoplist": list(STOP_WORDS)
    }
    kpm_weighting_params = {
        "df": cargo_df
    }
    kpm_extractor.extract(
        core_nlp_folder, n,
        selection_params=kpm_selection_params,
        weighting_params=kpm_weighting_params,
        output_file=os.path.join(EXTRACTED_DIR, "kpminer.csv")
    )
    # PKE: YAKE
    log.info("Begin extraction with a PKE extractor: YAKE")
    yake_extractor = extraction.PKEBasedExtractor(YAKE)
    yake_selection_params = {}
    yake_weighting_params = {
        "window": 2
    }
    yake_extractor.extract(
        core_nlp_folder, n,
        selection_params=yake_selection_params,
        weighting_params=yake_weighting_params,
        output_file=os.path.join(EXTRACTED_DIR, "yake.csv")
    )
    # PKE: SingleRank
    log.info("Begin Extraction with a PKE extractor: SingleRank")
    singlerank_extractor = extraction.PKEBasedExtractor(SingleRank)
    singlerank_selection_params = {
        "pos": considered_pos
    }
    singlerank_weighting_params = {
        "window": 10,
        "pos": considered_pos
    }
    singlerank_extractor.extract(
        core_nlp_folder, n,
        selection_params=singlerank_selection_params,
        weighting_params=singlerank_weighting_params,
        output_file=os.path.join(EXTRACTED_DIR, "singlerank.csv")
    )
    # PKE: TopicRank
    log.info("Begin Extraction with a PKE extractor: TopicRank")
    topicrank_extractor = extraction.PKEBasedExtractor(TopicRank)
    topicrank_selection_params = {
        "pos": considered_pos,
        "stoplist": list(STOP_WORDS)
    }
    topicrank_weighting_params = {}
    topicrank_extractor.extract(
        core_nlp_folder, n,
        selection_params=topicrank_selection_params,
        weighting_params=topicrank_weighting_params,
        output_file=os.path.join(EXTRACTED_DIR, "topicrank.csv")
    )
    # PKE: Multipartite
    log.info("Begin Extraction with a PKE extractor: MultipartiteRank")
    mprank_extractor = extraction.PKEBasedExtractor(MultipartiteRank)
    mprank_selection_params = {
        "pos": considered_pos,
        "stoplist": list(STOP_WORDS)
    }
    mprank_extractor.extract(
        core_nlp_folder, n,
        selection_params=mprank_selection_params,
        weighting_params={},
        output_file=os.path.join(EXTRACTED_DIR, "multipartite.csv")
    )
    # PKE: PositionRank
    log.info("Begin Extraction with a PKE extractor: PositionRank")
    positionrank_extractor = extraction.PKEBasedExtractor(PositionRank)
    positionrank_selection_params = {
        "grammar": r"""
                    NBAR:
                        {<NOUN|PROPN|NUM|ADJ>*<NOUN|PROPN>}

                    NP:
                        {<NBAR>}
                        {<NBAR><ADP><NBAR>}
                    """,
        "maximum_word_number": 5
    }
    positionrank_extractor.extract(
        core_nlp_folder, n,
        selection_params=positionrank_selection_params,
        weighting_params={},
        output_file=os.path.join(EXTRACTED_DIR, "positionrank.csv")
    )
    # EmbedRank
    log.info("Begin Extraction with EmbedRank extractor")
    embedrank_extractor = extraction.EmbedRankExtractor(
        emdib_model_path="pretrain_models/torontobooks_unigrams.bin"
    )
    embedrank_extractor.extract(
        core_nlp_folder, n,
        considered_tags=considered_pos,
        output_file=os.path.join(EXTRACTED_DIR, "embedrank_toronto_unigrams.csv")
    )


def evaluate_terms():
    annotated_corpus = corpus.Corpus(os.path.join(PROCESSED_DIR, "random_sample_annotated.xml"))
    log.info("Begin evaluation")
    evaluator = evaluation.Evaluator(annotated_corpus)
    extracted_terms = {
        "TF-IDF": "tfidf.csv",
        "KPM": "kpminer.csv",
        "YAKE": "yake.csv",
        "SingleRank": "singlerank.csv",
        "TopicRank": "topicrank.csv",
        "MultipartiteRank": "multipartite.csv",
        "PositionRank": "positionrank.csv",
        "EmbedRank": "embedrank_toronto_unigrams.csv"
    }
    for method, file_name in extracted_terms.items():
        terms = extraction.Extractor.read_terms_from(os.path.join(EXTRACTED_DIR, file_name))
        evaluator.add_prediction(method, terms)
    today_date = date.today().strftime("%Y%m%d")
    evaluator.evaluate_and_visualize(os.path.join(PLOT_DIR, f"eval_{today_date}.html"))


if __name__ == "__main__":
    scraping_news_sites()
    combine_filter_sample_corpus()
    manual_term_annotation()
    process_manual_annotation()
    create_core_nlp_documents(CORE_NLP_DIR)
    extract_terms(CORE_NLP_DIR)
    evaluate_terms()
