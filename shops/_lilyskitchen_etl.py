import asyncio
import json
import math
import pandas as pd
from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class LilysKitchenETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "LilysKitchen"
        self.BASE_URL = "https://www.lilyskitchen.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '.product-container'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        category_link = f"{self.BASE_URL}{category}"
        soup = asyncio.run(self.scrape(category_link, 'div#facet-main'))

        n_product = int(soup.find(
            'span', class_="product-facet__meta-bar-item--count").get_text().replace(' products', ''))
        n_pagination = math.ceil(n_product / 10)
        urls = [self.BASE_URL + product.get('href') for product in soup.find_all(
            'a', class_="product-item__aspect-ratio")]

        for i in range(1, n_pagination + 1):
            pagination_soup = asyncio.run(self.scrape(
                f"{category_link}?page={i}", 'div#facet-main', proxy=False, min_sec=0.5, max_sec=1))

            urls.extend([self.BASE_URL + product.get('href') for product in pagination_soup.find_all(
                'a', class_="product-item__aspect-ratio")])

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)
        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_name = soup.find(
                'h1', class_="product-meta__title").get_text()
            product_description = soup.find_all(
                'div', class_="product-tabs__item")[0].get_text()
            product_url = url.replace(self.BASE_URL, "")
            product_rating = '0/5'

            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            for variant in soup.find('ul', class_="bundled-variants__list").find_all('li'):
                variants.append(variant.find_all('span')[0].get_text())
                image_urls.append(
                    soup.find('meta', attrs={'property': 'og:image'}).get('content'))

                spans = variant.find_all('span')
                if len(spans) > 2:
                    discounted_price = float(variant.find_all(
                        'span')[1].get_text().replace('£', ''))
                    saving_price = float(variant.find_all(
                        'span')[2].get_text().split('£')[-1])
                    price = discounted_price + saving_price

                    discount_percentage = "{:.2f}".format(
                        (price - discounted_price) / price)

                    prices.append(price)
                    discounted_prices.append(discounted_price)
                    discount_percentages.append(discount_percentage)

                else:
                    price = float(variant.find_all('span')[1].get_text()[1:])
                    prices.append(price)
                    discounted_prices.append(None)
                    discount_percentages.append(None)

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
