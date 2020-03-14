import unittest
import preprocessing


class TestPreprocessing(unittest.TestCase):

    def test_combine_and_filter(self):
        combined_corpus = preprocessing.combine_xmls("../../data/test/scrape_samples/")
        self.assertEqual(len(combined_corpus), 102)
        filtered_corpus = preprocessing.filter_empty(combined_corpus)
        self.assertEqual(len(filtered_corpus), 99)


if __name__ == "__main__":
    unittest.main()
