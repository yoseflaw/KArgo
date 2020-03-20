import os
from nltk.corpus import stopwords
from kargo import logger, corpus, scraping, preprocessing, extraction, evaluation
from pke.unsupervised import PositionRank, MultipartiteRank
SCRAPED_DIR = "data/scraped/"
INTERIM_DIR = "data/interim/"
PROCESSED_DIR = "data/processed/"
MANUAL_DIR = "data/manual/"
EXTRACTED_DIR = "data/interim/extracted_terms"
PLOT_DIR = "plots/"
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
        seed_url="https://aircargoworld.com/allposts/category/news/page/",
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
    combined_corpus = preprocessing.combine_xmls(SCRAPED_DIR)
    log.info("Begin filtering empty documents")
    filtered_corpus = preprocessing.filter_empty(combined_corpus)
    n_sample = 10
    log.info(f"Begin sampling, n={n_sample}")
    sampled_corpus = filtered_corpus.get_sample(n_sample)
    log.info(f"Write sample.xml to {INTERIM_DIR}")
    sampled_corpus.write_xml_to(os.path.join(INTERIM_DIR, "sample.xml"))


def manual_term_annotation():
    log.info(f"Manual annotation assumed to be done, results in {MANUAL_DIR}")
    # currently done manually to output of combine_filter_sample_corpus
    # assumed result in MANUAL_DIR
    pass


def process_manual_annotation():
    log.info(f"Begin incorporating manual annotation to the XML, result in {PROCESSED_DIR}")
    manual_corpus = corpus.Corpus(os.path.join(MANUAL_DIR, "random_sample_annotated.xml"), is_annotated=True)
    manual_corpus.write_xml_to(os.path.join(PROCESSED_DIR, "random_sample_annotated.xml"))


def extract_terms():
    log.info("Begin Extraction")
    core_nlp_host = "localhost"
    core_nlp_port = 9000
    n = 15
    annotated_corpus = corpus.Corpus(os.path.join(PROCESSED_DIR, "random_sample_annotated.xml"))
    # prepare core nlp xmls
    core_nlp_folder = os.path.join(PROCESSED_DIR, "stanford_core_nlp_xmls")
    preprocessing.write_core_nlp_xmls(annotated_corpus, core_nlp_folder, host=core_nlp_host, port=core_nlp_port)
    # PKE: Multipartite
    log.info("Begin Extraction with a PKE extractor: MultipartiteRank")
    mprank_extractor = extraction.PKEBasedExtractor(MultipartiteRank)
    mprank_selection_params = {
        "pos": {"NOUN", "PROPN", "NUM", "ADJ", "ADP"},
        "stoplist": stopwords.words("english")
    }
    mprank_extractor.extract(core_nlp_folder, n, mprank_selection_params,
                             output_file=os.path.join(EXTRACTED_DIR, "multipartite.csv"))
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
    positionrank_extractor.extract(core_nlp_folder, n, positionrank_selection_params,
                                   output_file=os.path.join(EXTRACTED_DIR, "positionrank.csv"))
    # EmbedRank
    log.info("Begin Extraction with EmbedRank extractor")
    embedrank_extractor = extraction.EmbedRankExtractor(
        emdib_model_path="pretrain_models/torontobooks_unigrams.bin",
        core_nlp_host=core_nlp_host,
        core_nlp_port=core_nlp_port
    )
    embedrank_extractor.extract(annotated_corpus, n, output_file=os.path.join(EXTRACTED_DIR, "embedrank.csv"))


def evaluate_terms():
    annotated_corpus = corpus.Corpus(os.path.join(PROCESSED_DIR, "random_sample_annotated.xml"))
    log.info("Begin evaluation")
    evaluator = evaluation.Evaluator(annotated_corpus)
    multipartite_terms = extraction.Extractor.read_terms_from(os.path.join(EXTRACTED_DIR, "multipartite.csv"))
    evaluator.add_prediction("MultipartiteRank", multipartite_terms)
    positionrank_terms = extraction.Extractor.read_terms_from(os.path.join(EXTRACTED_DIR, "positionrank.csv"))
    evaluator.add_prediction("PositionRank", positionrank_terms)
    embedrank_terms = extraction.Extractor.read_terms_from(os.path.join(EXTRACTED_DIR, "embedrank.csv"))
    evaluator.add_prediction("EmbedRank", embedrank_terms)
    precision_scores = evaluator.calculate_precision_all()
    avg_precision_score = evaluation.Evaluator.get_average_scores(precision_scores)
    evaluation.Evaluator.visualize_scores(avg_precision_score, os.path.join(PLOT_DIR, "precision.html"))
    relative_recall_scores = evaluator.calculate_relative_recalls_all()
    avg_recall_score = evaluation.Evaluator.get_average_scores(relative_recall_scores)
    evaluation.Evaluator.visualize_scores(avg_recall_score, os.path.join(PLOT_DIR, "relative_recall.html"))


if __name__ == "__main__":
    scraping_news_sites()
    combine_filter_sample_corpus()
    manual_term_annotation()
    process_manual_annotation()
    extract_terms()
    evaluate_terms()
