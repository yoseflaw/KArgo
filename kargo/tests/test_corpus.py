import unittest
from kargo import corpus, logger
corpus.log.setLevel(logger.WARNING)


class TestCorpus(unittest.TestCase):

    def test_news(self):
        check_corpus1 = corpus.Corpus("../../data/test/samples_news_raw.xml")
        self.assertEqual(len(check_corpus1), 2)

    def test_annotated(self):
        check_corpus = corpus.Corpus(
            "../../data/test/samples_news_clean_unanno.xml",
            annotation_file="../../data/test/samples_with_manual_annotation.json1"
        )
        self.assertEqual(len(check_corpus), 2)

    def test_annotated_empty(self):
        check_corpus = corpus.Corpus(
            "../../data/test/samples_news_clean_random.xml",
            annotation_file="../../data/test/samples_with_manual_annotation.json1"
        )
        self.assertEqual(len(check_corpus), 0)

    def test_existing_annotation(self):
        check_corpus = corpus.Corpus("../../data/test/samples_with_terms.xml")
        self.assertEqual(len(check_corpus), 2)

    def test_existing_annotation_w_extra_annotation(self):
        check_corpus = corpus.Corpus(
            "../../data/test/samples_with_terms.xml",
            annotation_file="../../data/test/samples_with_manual_annotation.json1"
        )
        self.assertEqual(len(check_corpus), 2)

    def test_wiki(self):
        check_corpus = corpus.Corpus("../../data/test/samples_wiki.xml")
        self.assertEqual(len(check_corpus), 3)

    def test_sampling(self):
        check_corpus = corpus.Corpus("../../data/test/samples_news_clean_random.xml")
        sample_xml = check_corpus.get_sample(3)
        self.assertEqual(len(sample_xml), 3)

    def test_subsetting(self):
        check_corpus = corpus.Corpus("../../data/test/samples_news_clean_random.xml")
        sample_urls = [
            "https://theloadstar.com/will-digitisation-kill-off-forwarder/",
            "https://theloadstar.com/alert-airlines-unsafe-hoverboards-iata-calls-stiff-penalty-shippers-mis-declare-battery-devices/",
            "https://theloadstar.com/australian-box-terminal-operators-offset-falling-volumes-by-hiking-fees/",
        ]
        subset_xml = check_corpus.get_documents_by_urls(sample_urls)
        self.assertEqual(len(subset_xml), len(sample_urls))

    def test_combine_and_filter(self):
        combined_corpus = corpus.Corpus(xml_input="../../data/test/scrape_samples/")
        self.assertEqual(len(combined_corpus), 102)
        combined_corpus.filter_empty()
        self.assertEqual(len(combined_corpus), 99)


if __name__ == "__main__":
    unittest.main()
