import json
import asyncio
import pandas as pd

from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class TaylorPetFoodsETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "TaylorPetFoods"
        self.BASE_URL = "https://www.taylorspetfoods.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '.main-content'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        url = f"{self.BASE_URL}{category}"
        soup = asyncio.run(self.scrape(url, '#category-products'))

        if soup:
            urls = [self.BASE_URL + '/' + product.find('a').get('href')
                    for product in soup.find_all('div', class_="product-item")]
            df = pd.DataFrame({"url": urls})
            df.insert(0, "shop", self.SHOP)
            return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_name = soup.find(
                'div', class_="product-heading-d").find('h1').get_text()
            product_description = None

            if soup.find('div', id='tab-one'):
                product_description = soup.find(
                    'div', id='tab-one').find('span').get_text(strip=True)

            product_url = url.replace(self.BASE_URL, "")
            product_rating = '0/5'

            product_info = json.loads(
                soup.find('script', attrs={'type': 'application/ld+json'}).get_text())

            if isinstance(product_info, dict):
                product_info = [product_info]

            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            for variant in product_info:
                variants.append(variant.get('name').replace(
                    f"{product_name} - ", ''))
                prices.append(variant['offers']['price'])
                discounted_prices.append(None)
                discount_percentages.append(None)
                image_urls.append(
                    soup.find('meta', attrs={'property': "og:image"}).get('content'))

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
