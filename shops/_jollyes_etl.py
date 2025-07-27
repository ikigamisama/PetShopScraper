import asyncio
import pandas as pd
import json
import math
import re
from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class JollyesETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "Jollyes"
        self.BASE_URL = "https://www.jollyes.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#viewport'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 4

    def extract(self, category):
        category_link = f"{self.BASE_URL}/{category}.html"
        soup = asyncio.run(self.scrape(
            category_link, '#category', wait_until="networkidle"))

        subcategory_links = [link["href"] for ul in soup.select(
            "ul.second-category") for link in ul.select("a")]

        urls = []

        for subcategory in subcategory_links:
            url = self.BASE_URL + subcategory
            category_product_urls = []

            category_soup = asyncio.run(self.scrape(
                url, '.product-list', wait_until="networkidle", min_sec=3, max_sec=5))

            if not category_soup:
                logger.error(f"[ERROR] Failed to fetch or parse: {url}")
                continue

            sorting_row = category_soup.find('div', class_="sorting-row")
            if sorting_row:
                p_tag = sorting_row.find('p')
                if p_tag:
                    n_products_results = p_tag.get_text(strip=True)
                    n_products = int(re.findall(r'\d+', n_products_results)[0])
                else:
                    logger.warning(
                        f"[WARN] No <p> tag found in sorting row for {url}")
                    continue
            else:
                logger.warning(f"[WARN] No 'sorting-row' found for {url}")
                continue

            n_pagination = math.ceil(n_products / 100)

            product_tiles = category_soup.select("div[class*='product-tile']")
            category_product_urls = [
                self.BASE_URL + product_tile.find('a').get('href') for product_tile in product_tiles]

            if n_pagination > 1:
                for n in range(2, n_pagination + 1):
                    product_soup = asyncio.run(self.scrape(
                        f'{self.BASE_URL}{subcategory}?page={n}&perPage=100', '.product-list', wait_until="domcontentloaded", min_sec=3, max_sec=5))

                    if not product_soup:
                        logger.error(
                            f"[ERROR] Failed to fetch or parse: {self.BASE_URL}{subcategory}?page={n}&perPage=100")
                        continue

                    product_tiles = product_soup.select(
                        "div[class*='product-tile']")

                    category_product_urls = [
                        self.BASE_URL + product_tile.find('a').get('href') for product_tile in product_tiles]

            urls.extend(category_product_urls)

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)
        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            data = json.loads(soup.select_one(
                "section[class*='lazy-review-section']").select_one("script[type*='application']").text)
            product_title = data["name"]
            description = data["description"]

            if "aggregateRating" in data.keys():
                rating = data["aggregateRating"]["ratingCount"]
            else:
                rating = None

            product_url = url.replace(self.BASE_URL, "")
            price = float(data["offers"]["price"])

            df = pd.DataFrame(
                {
                    "shop": "Jollyes",
                    "name": product_title,
                    "rating": rating,
                    "description": description,
                    "url": product_url,
                    "price": price,
                    "image_urls": ', '.join(data['image']),
                    "variant": None,
                    "discounted_price": None,
                    "discount_percentage": None
                }, index=[0]
            )

            return df

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
