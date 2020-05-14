import os
import time
import random
import requests
from datetime import date, datetime
from bs4 import BeautifulSoup
from kargo.corpus import Corpus
from kargo import logger
log = logger.get_logger(__name__, logger.INFO)


class Spider(object):

    def __init__(self, seed_url, output_folder):
        self.seed_url = seed_url
        self.output_folder = output_folder
        self.current_page = 0

    @staticmethod
    def get_soup(url):
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:53.0) Gecko/20100101 Firefox/53.0"
        }
        page = requests.get(url, headers=headers)
        if page.status_code != 200:
            print(f"[ERROR] Status Code={page.status_code} for URL={url}")
            return None
        soup = BeautifulSoup(page.content, "html.parser")
        _ = [s.extract() for s in soup("script")]
        return soup

    def start(self, begin_page, end_page):
        for page_no in range(begin_page, end_page):
            self.scrape(page_no)

    def scrape(self, page_no):
        self.current_page = page_no
        soup = Spider.get_soup(self.seed_url + str(page_no))
        log.info(f"Extracting Page-{page_no}")
        page_corpus = self.extract_pages(soup)
        today_date = date.today().strftime("%Y%m%d")
        page_corpus.write_xml_to(os.path.join(self.output_folder, f"{today_date}_{self.current_page}.xml"))
        return page_corpus

    def extract_pages(self, soup):
        pass


class AirCargoNewsSpider(Spider):

    def extract_pages(self, soup):
        corpus = Corpus()
        news_snippets = soup.find_all("div", class_="post-details")
        for news_snippet in news_snippets:
            time.sleep(random.randint(1, 3))
            # get news metadata
            url = news_snippet.find("a")["href"]
            log.info(f"Scraping URL={url}")
            if len(url.split("/")) <= 5: continue
            title = news_snippet.find("h3", class_="post-title").text
            categories = url.split("/")[3]
            author_p = news_snippet.find("p", class_="post-thumb-author")
            author = author_p.text[3:] if author_p is not None else ""
            # get news content
            news_soup = Spider.get_soup(url)
            if not news_soup: continue
            paragraphs = news_soup.find("div", class_="content-section clearfix")
            if paragraphs is None: continue
            paragraphs = paragraphs.find_all("p")
            published_time = news_soup.find("meta", attrs={"property": "article:published_time"})["content"]
            content = [paragraph.text for paragraph in paragraphs]
            # get news tags
            tag_links = news_soup.find("li", class_="tag")
            tags = [link.text for link in tag_links.find_all("a")] if tag_links is not None else ""
            # put into news object
            corpus.add_document(url, title, categories, published_time, content, author, tags)
        return corpus


class AirCargoWeekSpider(Spider):

    def extract_pages(self, soup):
        corpus = Corpus()
        news_snippets = soup.find("div", class_="td-ss-main-content").find_all("h3", class_="entry-title")
        for news_snippet in news_snippets:
            time.sleep(random.randint(1, 3))
            url = news_snippet.find("a")["href"]
            log.info(f"Scraping URL={url}")
            news_soup = Spider.get_soup(url)
            if not news_soup: continue
            title = news_soup.find("h1", class_="entry-title").text
            categories = news_soup.find("div", class_="td-crumb-container").find_all("span")[1].find("a").text
            author = news_soup.find("div", class_="td-post-author-name").find("a").text
            published_time = news_soup.find("meta", attrs={"property": "article:published_time"})["content"]
            paragraphs = news_soup.find("div", class_="td-post-content").find_all("p")
            content = [paragraph.text.strip() for paragraph in paragraphs if len(paragraph.text.strip()) > 0]
            # get news tags
            tag_metas = news_soup.find_all("meta", attrs={"property": "article:tag"})
            tags = [tag_meta["content"] for tag_meta in tag_metas] if tag_metas is not None else ""
            corpus.add_document(url, title, categories, published_time, content, author, tags)
        return corpus


class AirCargoWorldSpider(Spider):

    def extract_pages(self, soup):
        corpus = Corpus()
        news_snippets = soup.find("div", id="rd-blog").find_all("h3", class_="rd-title")
        for news_snippet in news_snippets:
            time.sleep(random.randint(1, 3))
            url = news_snippet.find("a")["href"]
            log.info(f"Scraping URL={url}")
            if "aircargoworld.com" not in url: continue
            news_soup = Spider.get_soup(url)
            if not news_soup or news_soup.find("input", id="wp-submit"): continue
            paragraphs = news_soup.find("article", class_="rd-post-content").find_all("p")
            content = []
            for paragraph in paragraphs:
                paragraph_text = paragraph.text.strip()
                if len(paragraph_text) > 0:
                    content.append(paragraph_text)
            title = news_soup.find("h3", class_="rd-title entry-title").text
            categories = news_soup.find("li", class_="rd-cats").find_all("a")
            categories = [topic.text for topic in categories]
            author = news_soup.find("li", class_="rd-author").find("a").text
            published_time = news_soup.find("li", class_="rd-date")
            if published_time is not None:
                published_time = datetime.strptime(published_time.text, "%B %d, %Y")
                published_time = published_time.strftime("%Y-%m-%d")  # assumed
            else:
                published_time = ""
            tag_uls = news_soup.find("ul", class_="rd-tags")
            if tag_uls is not None:
                tag_uls = tag_uls.find_all("a")
                tags = [tag_ul.text for tag_ul in tag_uls if tag_ul.text != "Tags"]
            else:
                tags = []
            corpus.add_document(url, title, categories, published_time, content, author, tags)
        return corpus
    

class StatTimesSpider(Spider):
    
    def extract_pages(self, soup):
        corpus = Corpus()
        news_snippets = soup.find_all("h4", class_="post-title-new")
        for news_snippet in news_snippets:
            time.sleep(random.randint(1, 3))
            url = news_snippet.find("a")["href"]
            log.info(f"Scraping URL={url}")
            news_soup = Spider.get_soup(url)
            if not news_soup: continue
            title = news_soup.find("div", class_="single-header").find("h1").text
            categories = ["Air Cargo"]
            author = news_soup.find("meta", attrs={"name": "author"})["content"]
            published_time_meta = news_soup.find("meta", attrs={"article:published_time"})
            if published_time_meta:
                published_time = published_time_meta["content"]
            else:
                updated_time_meta = news_soup.find("meta", attrs={"property": "og:updated_time"})
                if updated_time_meta:
                    published_time = updated_time_meta["content"]
                else:
                    published_time_raw = news_soup.find("span", class_="meta_date")
                    published_time = datetime.strptime(published_time_raw.text, "%B %d, %Y")
                    published_time = published_time.strftime("%Y-%m-%d")  # assumed
            paragraphs = news_soup.find("div", class_="single-content").find_all(
                "p", attrs={"data-mce-style": "text-align: justify;"})
            content = []
            for paragraph in paragraphs:
                paragraph_text = paragraph.text
                if len(content) == 0:
                    # remove date from the news
                    paragraph_text = paragraph_text[paragraph_text.find(":")+1:]
                if len(paragraph_text.strip()) > 0:
                    content.append(paragraph_text.strip())
            tag_metas = news_soup.find("meta", attrs={"name": "news_keywords"})["content"]
            tags = [keyword.strip() for keyword in tag_metas.split(",") if len(keyword.strip()) > 0]
            corpus.add_document(url, title, categories, published_time, content, author, tags)
        return corpus


class TheLoadStarSpider(Spider):

    def extract_pages(self, soup):
        corpus = Corpus()
        news_snippets = soup.find_all("article", class_="card pad")
        for news_snippet in news_snippets:
            time.sleep(random.randint(1, 3))
            url = news_snippet.find("a")["href"]
            log.info(f"Scraping URL={url}")
            if "theloadstar.com" not in url: continue
            news_soup = Spider.get_soup(url)
            if not news_soup: continue
            paragraphs = news_soup.find("article").find_all("p")
            content = [paragraph.text.strip() for paragraph in paragraphs if len(paragraph.text.strip()) > 0]
            title = news_soup.find("section", class_="single-article").find("a").text
            categories = news_soup.find("meta", attrs={"property": "article:section"})
            categories = [categories["content"]] if categories else None
            author = news_soup.find("address", class_="author")
            author = author.find("a").text if author else ""
            published_time = news_soup.find("meta", attrs={"property": "article:published_time"})
            published_time = published_time["content"] if published_time else ""
            tag_metas = news_soup.find_all("meta", attrs={"property": "article:tag"})
            tags = [tag_meta["content"] for tag_meta in tag_metas]
            corpus.add_document(url, title, categories, published_time, content, author, tags)
        return corpus
