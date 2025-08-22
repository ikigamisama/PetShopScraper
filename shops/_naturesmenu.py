import re
import math
import asyncio
import pandas as pd


from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class NaturesMenuETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "NaturesMenu"
        self.BASE_URL = "https://www.naturesmenu.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#maincontent'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    import re

    def extract(self, category):
        url = self.BASE_URL + category
        soup = asyncio.run(self.scrape(url, '#maincontent'))

        counter_text = soup.find(
            'div', id="search-result-counter-sm").get_text(strip=True)
        match = re.search(r'\d+', counter_text)
        if not match:
            logger.warning(
                f"[WARN] Could not parse product count for {category}. Raw counter: '{counter_text}'")
            return pd.DataFrame(columns=["shop", "url"])

        n_product = int(match.group())
        if n_product == 0:
            logger.info(f"[INFO] No products found for category: {category}")
            return pd.DataFrame(columns=["shop", "url"])

        pagination_length = math.ceil(n_product / 12)
        urls = []

        for i in range(1, pagination_length + 1):
            soup_pagination = asyncio.run(
                self.scrape(f"{url}?page={i}", '#maincontent'))
            try:
                grid = soup_pagination.find('div', class_="product-grid")
                products = grid.find_all('div', class_="product")
            except AttributeError:
                logger.error(
                    f"[ERROR] Page structure invalid on page {i} of {category}")
                continue

            for prod in products:
                href = prod.find('a').get('href')
                if href:
                    urls.append(self.BASE_URL + href)

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)
        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_name = soup.find(
                'h2', class_="product-type").get_text() + ' ' + soup.find('h1', class_="name").get_text()
            product_description = None

            if soup.find('div', class_="description"):
                product_description = soup.find(
                    'div', class_="description").find('p').get_text()

            product_url = url.replace(self.BASE_URL, "")
            product_rating = '0/5'

            if soup.find('div', class_="pdp-feefo-product-reviews-summary-rating-border"):
                product_rating = soup.find(
                    'div', class_="pdp-feefo-product-reviews-summary-rating-border").find('p').get_text(strip=True) + "/5"

            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            price_info = soup.find('button', class_="add-to-cart")

            if price_info.get('data-item-id-bundle') == 'null':
                variants.append(price_info.get('data-item-variant'))
                image_urls.append(
                    soup.find('meta', attrs={'property': "og:image"}).get('content'))
                prices.append(price_info.get('data-item-price'))
                discounted_prices.append(None)
                discount_percentages.append(None)
            else:
                variants = [price_info.get(
                    'data-item-variant'), price_info.get('data-item-variant-bundle')]
                prices = [price_info.get(
                    'data-item-price'), price_info.get('data-item-price')]
                image_urls = [soup.find('meta', attrs={'property': "og:image"}).get(
                    'content'), soup.find('meta', attrs={'property': "og:image"}).get('content')]
                discounted_prices = [None, None]
                discount_percentages = [None, None]

            df = pd.DataFrame({
                "variant": variants,
                "price": prices,
                "discounted_price": discounted_prices,
                "discount_percentage": discount_percentages,
                "image_urls": image_urls
            })
            df.insert(0, "url", product_url)
            df.insert(0, "description", product_description)
            df.insert(0, "rating", product_rating)
            df.insert(0, "name", product_name)
            df.insert(0, "shop", self.SHOP)

            return df
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
