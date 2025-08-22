import pandas as pd
import asyncio
import math
import re
from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class HarringtonsETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "Harringtons"
        self.BASE_URL = "https://www.harringtonspetfood.com"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#MainContent'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        category_link = f"{self.BASE_URL}{category}"

        urls = []
        soup = asyncio.run(self.scrape(category_link, '#MainContent'))

        n_product = int(soup.find(
            'span', class_="boost-pfs-filter-total-product").find(string=True, recursive=False))
        pagination_length = math.ceil(n_product / 24)

        for i in range(1, pagination_length + 1):
            soup_pagination = asyncio.run(self.scrape(
                f"{category_link}?page={i}", '#MainContent'))
            for prod_list in soup_pagination.find_all('li', class_="list-product-card__item"):
                urls.append(self.BASE_URL + prod_list.find('a',
                            class_="card-product__heading-link").get('href').replace('#', ''))

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)
        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_name = soup.find(
                'h1', class_="header-product__heading").get_text()

            product_description = None

            if soup.find('div', class_="panel-product-description__single-content"):
                product_description = soup.find(
                    'div', class_="panel-product-description__single-content").get_text()
            else:
                product_description = soup.find(
                    'div', class_="panel-product-description__copy").get_text()

            product_url = url.replace(self.BASE_URL, "")
            product_rating = re.sub(r'[^\d.]', '', soup.find(
                'div', class_="okeReviews-reviewsSummary-starRating").find('span', class_="okeReviews-a11yText").get_text()) + '/5'

            variants = [None]
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            price_container = soup.find('div', class_="price__container")
            original_price = None
            sale_price = None
            discount = None

            savings_elem = price_container.select_one(
                ".sale-item-savings-amount")

            if savings_elem:
                savings_text = re.sub(
                    r'[^\d.-]', '', savings_elem.text.strip())
                try:
                    savings_value = float(savings_text) if savings_text else 0
                except ValueError:
                    savings_value = 0

                has_discount = savings_value > 0
            else:
                has_discount = False

            if has_discount:

                original_price_elem = price_container.select_one(
                    ".sale-compare-amounts s.price-item--regular")
                original_price = original_price_elem.text.strip().replace(
                    "RRP:", "").strip() if original_price_elem else None

                sale_price_elem = price_container.select_one(
                    ".price__sale .price-item--sale")
                sale_price = sale_price_elem.contents[0].strip().replace(
                    '£', '') if sale_price_elem else None

                discount_elem = price_container.select_one(
                    ".sale-item-discount-amount")
                discount = discount_elem.text.strip().replace(
                    '% off', '') if discount_elem else None
                discount = int(discount)/100

            else:
                no_discount_price_elem = price_container.select_one(
                    ".price__regular .price-item--regular")
                original_price = no_discount_price_elem.text.strip().replace(
                    "RRP", "").strip() if no_discount_price_elem else None

            prices.append(original_price.replace('£', ''))
            discounted_prices.append(sale_price)
            discount_percentages.append(discount)
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
