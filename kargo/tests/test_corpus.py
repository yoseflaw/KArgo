import unittest
from kargo import corpus, logger
corpus.log.setLevel(logger.WARNING)


class TestCorpus(unittest.TestCase):

    def test_news(self):
        check_corpus1 = corpus.Corpus("../../data/test/samples_news_1.xml")
        self.assertEqual(len(check_corpus1), 2)
        check_corpus2 = corpus.Corpus("../../data/test/samples_news_2.xml")
        self.assertEqual(len(check_corpus2), 102)

    def test_annotated(self):
        check_corpus = corpus.Corpus("../../data/test/samples_with_manual_annotation.xml", is_annotated=True)
        self.assertEqual(len(check_corpus), 3)

    def test_wiki(self):
        check_corpus = corpus.Corpus("../../data/test/samples_wiki.xml")
        self.assertEqual(len(check_corpus), 3)

    def test_sampling(self):
        check_corpus = corpus.Corpus("../../data/test/samples_news_2.xml")
        sample_xml = check_corpus.get_sample(10)
        self.assertEqual(len(sample_xml), 10)

    def test_subsetting(self):
        check_corpus = corpus.Corpus("../../data/test/samples_news_2.xml")
        sample_urls = [
            "https://aircargoworld.com/allposts/delta-american-cutbacks-add-to-global-airline-meltdown-on-virus/",
            "https://www.aircargoweek.com/aeroflot-adjusts-flight-schedule-for-italy-france-germany-and-spain/",
            "https://www.stattimes.com/news/b-h-worldwide-to-handle-aog-technics-logistics-in-germany/",
            "https://theloadstar.com/coronavirus-impact-subsiding-rapidly-as-china-shipping-revives/",
            "https://www.aircargonews.net/freight-forwarder/damco-launches-three-tier-product-offering/"
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
