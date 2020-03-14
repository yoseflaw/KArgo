import unittest
from pipeline import scraping


class TestScraping(unittest.TestCase):

    def test_aircargonews(self):
        air_cargo_news_spider = scraping.AirCargoNewsSpider(
            seed_url="https://www.aircargonews.net/news-by-date/page/",
            output_folder="../data/interim/"
        )
        result = air_cargo_news_spider.scrape(1)
        self.assertGreater(len(result), 0)

    def test_aircargoweek(self):
        air_cargo_week_spider = scraping.AirCargoWeekSpider(
            seed_url="https://www.aircargoweek.com/category/news-menu/page/",
            output_folder="../data/interim/"
        )
        result = air_cargo_week_spider.scrape(1)
        self.assertGreater(len(result), 0)

    def test_aircargoworld(self):
        air_cargo_world_spider = scraping.AirCargoWorldSpider(
            seed_url="https://aircargoworld.com/allposts/category/news/page/",
            output_folder="../data/interim/"
        )
        result = air_cargo_world_spider.scrape(1)
        self.assertGreater(len(result), 0)

    def test_stattimes(self):
        stat_times_spider = scraping.StatTimesSpider(
            seed_url="https://www.stattimes.com/category/air-cargo/page/",
            output_folder="../data/interim/"
        )
        result = stat_times_spider.scrape(1)
        self.assertGreater(len(result), 0)

    def test_theloadstar(self):
        the_load_star_spider = scraping.TheLoadStarSpider(
            seed_url="https://theloadstar.com/category/news/page/",
            output_folder="../data/interim/"
        )
        result = the_load_star_spider.scrape(1)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
