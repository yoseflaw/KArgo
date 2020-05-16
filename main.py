import os
from datetime import date
from spacy.lang.en.stop_words import STOP_WORDS
from kargo import logger, corpus, scraping, terms, evaluation
from pke.utils import compute_document_frequency, load_document_frequency_file
from pke.unsupervised import TfIdf, KPMiner, YAKE
from pke.unsupervised import SingleRank, TopicRank, PositionRank, MultipartiteRank
SCRAPED_DIR = "data/scraped"
INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"
MANUAL_DIR = "data/manual"
RESULTS_DIR = "results"
CORE_NLP_DIR = os.path.join(PROCESSED_DIR, "scnlp_xmls")
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
        os.path.join(INTERIM_DIR, "lda_sampling_15p.xml"),
        annotation_file=os.path.join(MANUAL_DIR, "annotation.json1")
    )
    manual_corpus.write_xml_to(os.path.join(PROCESSED_DIR, "lda_sampling_15p.annotated.xml"))


def create_core_nlp_documents(core_nlp_folder):
    log.info(f"Begin preparing Core NLP Documents to {core_nlp_folder}")
    annotated_corpus = corpus.Corpus(os.path.join(PROCESSED_DIR, "lda_sampling_15p.annotated.xml"))
    annotated_corpus.write_to_core_nlp_xmls(core_nlp_folder)
    compute_document_frequency(
        core_nlp_folder, os.path.join(INTERIM_DIR, "cargo_df.tsv.gz"),
        stoplist=list(STOP_WORDS)
    )


# noinspection PyTypeChecker
def extract_terms(core_nlp_folder):
    log.info("Begin Extraction")
    n = 10
    cargo_df = load_document_frequency_file(os.path.join(INTERIM_DIR, "cargo_df.tsv.gz"))
    pke_factory = {
        "grammar": r"""
            NBAR:
                {<NOUN|PROPN|NUM|ADJ>*<NOUN|PROPN>}
    
            NP:
                {<NBAR>}
                {<NBAR><ADP><NBAR>}
            """,
        "filtering_params": {
            "stoplist": list(STOP_WORDS)
        },
        "extractors": {
            "tfidf": {
                "instance": terms.PKEBasedTermsExtractor(TfIdf),
                "weighting_params": {"df": cargo_df}
            },
            "kpm": {
                "instance": terms.PKEBasedTermsExtractor(KPMiner),
                "weighting_params": {"df": cargo_df}
            },
            "yake": {
                "instance": terms.PKEBasedTermsExtractor(YAKE),
                "filtering_params": {
                    "only_alphanum": True,
                    "strip_outer_stopwords": True
                },
                "weighting_params": {}
            },
            "singlerank": {
                "instance": terms.PKEBasedTermsExtractor(SingleRank),
                "weighting_params": {
                    "window": 10,
                    "pos": {"NOUN", "PROPN", "NUM", "ADJ", "ADP"}
                }
            },
            "topicrank": {
                "instance": terms.PKEBasedTermsExtractor(TopicRank),
                "weighting_params": {}
            },
            "mprank": {
                "instance": terms.PKEBasedTermsExtractor(MultipartiteRank),
                "weighting_params": {}
            },
            "positionrank": {
                "instance": terms.PKEBasedTermsExtractor(PositionRank),
                "weighting_params": {}
            }
        }
    }
    for name in pke_factory["extractors"]:
        log.info(f"Begin Extraction with PKE based extractor: {name}")
        extractor = pke_factory["extractors"][name]["instance"]
        if "filtering_params" in pke_factory["extractors"][name]:
            filtering_params = {
                **pke_factory["filtering_params"],
                **pke_factory["extractors"][name]["filtering_params"]
            }
        else:
            filtering_params = pke_factory["filtering_params"]
        extractor.extract(
            core_nlp_folder, n,
            grammar=pke_factory["grammar"],
            filtering_params=filtering_params,
            weighting_params=pke_factory["extractors"][name]["weighting_params"],
            output_file=os.path.join(EXTRACTED_DIR, f"{name}.csv")
        )
    # EmbedRank
    log.info("Begin Extraction with EmbedRank extractor")
    embedrank_extractor = terms.EmbedRankTermsExtractor(
        emdib_model_path="pretrain_models/torontobooks_unigrams.bin"
    )
    embedrank_extractor.extract(
        core_nlp_folder, n,
        grammar=r"""
                NALL:
                    {<NN|NNP|NNS|NNPS>}

                NBAR:
                    {<NALL|CD|JJ>*<NALL>}

                NP:
                    {<NBAR>}
                    {<NBAR><IN><NBAR>}
                """,
        considered_tags={'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'IN', 'CD'},
        output_file=os.path.join(EXTRACTED_DIR, "embedrank_toronto_unigrams.csv")
    )


def evaluate_terms():
    annotated_corpus = corpus.Corpus(os.path.join(PROCESSED_DIR, "lda_sampling_15p.annotated.xml"))
    log.info("Begin evaluation")
    evaluator = evaluation.Evaluator(annotated_corpus)
    extracted_terms = {
        "TF-IDF": "tfidf.csv",
        "KPM": "kpm.csv",
        "YAKE": "yake.csv",
        "SingleRank": "singlerank.csv",
        "TopicRank": "topicrank.csv",
        "MultipartiteRank": "mprank.csv",
        "PositionRank": "positionrank.csv",
        "EmbedRank": "embedrank_toronto_unigrams.csv"
    }
    for method, file_name in extracted_terms.items():
        t = terms.TermsExtractor.read_terms_from(os.path.join(EXTRACTED_DIR, file_name))
        evaluator.add_prediction(method, t)
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
