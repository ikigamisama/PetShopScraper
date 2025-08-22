import math
import asyncio
import pandas as pd
from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class PurinaETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "Purina"
        self.BASE_URL = "https://www.purina.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#main-layout'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        current_url = f"{self.BASE_URL}{category}"
        urls = []

        soup = asyncio.run(self.scrape(current_url, '.main-view-content'))

        if soup:
            n_product = int(soup.find(
                'div', class_="view-header").find('div', class_="header").get_text().split(' of ')[1])
            n_pagination = math.ceil(n_product / 12)

            urls = [self.BASE_URL + product.get('href') for product in soup.find_all(
                'a', class_="product-tile_image")]
            for n in range(1, n_pagination + 1):
                pagination_url = current_url + f'?page={n}'
                pagination_soup = asyncio.run(
                    self.scrape(pagination_url, '.main-view-content'))

                urls.extend([self.BASE_URL + product.get('href') for product in pagination_soup.find_all(
                    'a', class_="product-tile_image")])

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)

        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_name = soup.find(
                'h1', class_="dsu-product--title").get_text(strip=True)
            product_url = url.replace(self.BASE_URL, "")

            product_description_meta = soup.find(
                'meta', attrs={'property': 'og:description'})

            if product_description_meta:
                product_description = product_description_meta.get('content')
            else:
                fallback_meta = soup.find(
                    'meta', attrs={'name': 'description'})
                product_description = fallback_meta.get(
                    'content') if fallback_meta else None

            product_rating = '0/5'

            rating_wrapper = soup.find(
                'div', attrs={'class': ['review-stats test1']})
            if rating_wrapper:
                product_rating = rating_wrapper.find(
                    'div', class_='count').getText(strip=True)

            variants = [None]
            prices = [None]
            discounted_prices = [None]
            discount_percentages = [None]

            image_urls = [', '.join([self.BASE_URL + img.find('img').get('src') for img in soup.find(
                'div', class_="carousel-media").find_all('div', class_="field__item")])]
            df = pd.DataFrame(
                {
                    "variant": variants,
                    "price": prices,
                    "discounted_price": discounted_prices,
                    "discount_percentage": discount_percentages,
                    "image_urls": image_urls
                }
            )

            df.insert(0, "url", product_url)
            df.insert(0, "description", product_description)
            df.insert(0, "rating", product_rating)
            df.insert(0, "name", product_name)
            df.insert(0, "shop", self.SHOP)

            return df

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
