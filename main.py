import os
from spacy.lang.en.stop_words import STOP_WORDS
from kargo import logger, corpus, scraping, extraction, evaluation
from pke.unsupervised import PositionRank, MultipartiteRank, EmbedRank
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


def create_core_nlp_documents(core_nlp_folder):
    log.info("Begin preparing Core NLP Documents")
    annotated_corpus = corpus.Corpus(os.path.join(PROCESSED_DIR, "random_sample_annotated.xml"))
    annotated_corpus.write_to_core_nlp_xmls(core_nlp_folder)


def extract_terms(core_nlp_folder):
    log.info("Begin Extraction")
    n = 15
    considered_pos = {"NOUN", "PROPN", "NUM", "ADJ", "ADP"}
    # PKE: Multipartite
    # log.info("Begin Extraction with a PKE extractor: MultipartiteRank")
    # mprank_extractor = extraction.PKEBasedExtractor(MultipartiteRank)
    # mprank_selection_params = {
    #     "pos": considered_pos,
    #     "stoplist": list(STOP_WORDS)
    # }
    # mprank_extractor.extract(core_nlp_folder, n,
    #                          selection_params=mprank_selection_params,
    #                          weighting_params={},
    #                          output_file=os.path.join(EXTRACTED_DIR, "multipartite.csv"))
    # # PKE: PositionRank
    # log.info("Begin Extraction with a PKE extractor: PositionRank")
    # positionrank_extractor = extraction.PKEBasedExtractor(PositionRank)
    # positionrank_selection_params = {
    #     "grammar": r"""
    #                 NBAR:
    #                     {<NOUN|PROPN|NUM|ADJ>*<NOUN|PROPN>}
    #
    #                 NP:
    #                     {<NBAR>}
    #                     {<NBAR><ADP><NBAR>}
    #                 """,
    #     "maximum_word_number": 5
    # }
    # positionrank_extractor.extract(core_nlp_folder, n,
    #                                selection_params=positionrank_selection_params,
    #                                weighting_params={},
    #                                output_file=os.path.join(EXTRACTED_DIR, "positionrank.csv"))
    # EmbedRank
    log.info("Begin Extraction with EmbedRank extractor")
    embedrank_extractor = extraction.EmbedRankExtractor(
        emdib_model_path="pretrain_models/wiki_unigrams.bin"
    )
    embedrank_extractor.extract(
        core_nlp_folder, n,
        considered_tags=considered_pos,
        output_file=os.path.join(EXTRACTED_DIR, "embedrank_wiki_unigrams.csv")
    )


def evaluate_terms():
    annotated_corpus = corpus.Corpus(os.path.join(PROCESSED_DIR, "random_sample_annotated.xml"))
    log.info("Begin evaluation")
    evaluator = evaluation.Evaluator(annotated_corpus)
    multipartite_terms = extraction.Extractor.read_terms_from(os.path.join(EXTRACTED_DIR, "multipartite.csv"))
    evaluator.add_prediction("MultipartiteRank", multipartite_terms)
    positionrank_terms = extraction.Extractor.read_terms_from(os.path.join(EXTRACTED_DIR, "positionrank.csv"))
    evaluator.add_prediction("PositionRank", positionrank_terms)
    embedrank_wiki_terms = extraction.Extractor.read_terms_from(
        os.path.join(EXTRACTED_DIR, "embedrank_wiki_unigrams.csv")
    )
    evaluator.add_prediction("EmbedRank_wiki", embedrank_wiki_terms)
    precision_scores = evaluator.calculate_precision_all()
    avg_precision_score = evaluation.Evaluator.get_average_scores(precision_scores)
    evaluation.Evaluator.visualize_scores(avg_precision_score, os.path.join(PLOT_DIR, "precision.html"))
    relative_recall_scores = evaluator.calculate_relative_recalls_all()
    avg_recall_score = evaluation.Evaluator.get_average_scores(relative_recall_scores)
    evaluation.Evaluator.visualize_scores(avg_recall_score, os.path.join(PLOT_DIR, "relative_recall.html"))


if __name__ == "__main__":
    # scraping_news_sites()
    # combine_filter_sample_corpus()
    # manual_term_annotation()
    # process_manual_annotation()
    core_nlp_input = os.path.join(PROCESSED_DIR, "stanford_core_nlp_xmls")
    # create_core_nlp_documents(core_nlp_input)
    # extract_terms(core_nlp_input)
    evaluate_terms()
