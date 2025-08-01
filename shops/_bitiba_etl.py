import re
import json
import math
import time
import random
import pandas as pd
import requests

from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log
MIN_WAIT_BETWEEN_REQ = 10
MAX_WAIT_BETWEEN_REQ = 15
MAX_RETRIES = 5


class ScrapingError(Exception):
    pass


class BitibaETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "Bitiba"
        self.BASE_URL = "https://www.bitiba.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = 'main#page-content'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 3
        self.with_proxy = True

    @retry(
        wait=wait_exponential(
            multiplier=1, min=MIN_WAIT_BETWEEN_REQ, max=MAX_WAIT_BETWEEN_REQ),
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception_type(ScrapingError),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    def _fetch_json_with_retry(self, url):
        response = requests.get(url)
        if response.status_code != 200:
            raise ScrapingError(
                f"Failed to fetch: {url} | Status: {response.status_code}")
        try:
            return response.json()
        except Exception as e:
            raise ScrapingError(f"Failed to parse JSON from {url}: {e}")

    def extract(self, category):
        urls = []
        base_api_url = (
            "https://www.bitiba.co.uk/api/discover/v1/products/list-faceted-partial"
            "?&path={category}&domain=bitiba.co.uk&language=en&page={page}&size=24"
            "&ab=shop-10734_shop_product_catalog_api_enabled_targeted_delivery.enabled"
            "%2Bidpo-1141_article_based_product_cards_targeted_delivery.on"
            "%2Bshop-11393_disable_plp_spc_api_cache_targeted_delivery.on"
            "%2Bshop-11371_enable_sort_by_unit_price_targeted_delivery.on"
            "%2Bidpo-1390_rebranding_foundation_targeted_delivery.on"
            "%2Bexplore-3092-price-redesign_targeted_delivery.on"
        )

        def build_url(page):
            return base_api_url.format(category=category, page=page)

        first_url = build_url(1)
        logger.info(f"Accessing: {first_url}")

        try:
            product_data = self._fetch_json_with_retry(first_url)
        except ScrapingError as e:
            logger.error(str(e))
            return pd.DataFrame(columns=["shop", "url"])

        pagination = product_data.get("pagination")
        if not isinstance(pagination, dict):
            logger.error(
                "'pagination' is missing or not a dict in response JSON.")

            products = product_data.get('productList', {}).get('products', [])
            urls.extend([
                self.BASE_URL.rstrip('/') + product['path']
                for product in products
                if product.get('path')
            ])

            logger.info(
                f"Extracted {len(urls)} product URLs from fallback (no pagination).")
            df = pd.DataFrame({"url": urls})
            df.insert(0, "shop", self.SHOP)
            return df

        n_pagination = pagination.get('count', 0)
        n_products_text = product_data['productList']["productListHeading"]["totalProductsText"]
        n_products = int(re.search(r'of (\d+)', n_products_text).group(1))

        logger.info(
            f"Found {n_products} products across {n_pagination} pages.")

        time.sleep(random.uniform(10, 15))

        for page in range(1, n_pagination + 1):
            page_url = build_url(page)
            logger.info(f"Accessing page {page}: {page_url}")

            try:
                data_product = self._fetch_json_with_retry(page_url)
                products = data_product.get(
                    'productList', {}).get('products', [])
                urls.extend([
                    self.BASE_URL.rstrip('/') + product['path']
                    for product in products
                    if product.get('path')
                ])
            except ScrapingError as e:
                logger.warning(f"Skipping page {page}: {str(e)}")
                continue

            time.sleep(random.uniform(10, 15))

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)
        logger.info(f"Total extracted URLs: {len(df)}")
        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_data_list = soup.select(
                "script[type*='application/ld+json']")
            if product_data_list:
                product_data = json.loads(product_data_list[1].text)

                product_title = product_data["name"]
                rating = '0/5'
                if "aggregateRating" in product_data.keys():
                    rating = product_data["aggregateRating"]["ratingValue"]
                    rating = f"{rating}/5"

                description = product_data["description"]
                product_url = url.replace(self.BASE_URL, "")

                # Placeholder for variant details
                variants = []
                prices = []
                discounted_prices = []
                discount_percentages = []
                image_urls = []

                pattern = r"^.*£"
                rrb_pattern = r"[^\d\.]"

                variants_list = soup.find(
                    'div', class_="VariantList_variantList__PeaNd")
                if variants_list:
                    variant_hopps = variants_list.select(
                        "div[data-hopps*='Variant']")
                    for variant_hopp in variant_hopps:

                        variant = variant_hopp.select_one(
                            "span[class*='VariantDescription_description']").text
                        image_variant = variant_hopp.find('img').get('src')
                        discount_checker = variant_hopp.find(
                            'div', class_="z-product-price__note-wrap")

                        if discount_checker:
                            price = float(re.sub(rrb_pattern, "", variant_hopp.select_one(
                                "div[class*='z-product-price__nowrap']").text))
                            discounted_price = float(re.sub(pattern, "", variant_hopp.select_one(
                                "span[class*='z-product-price__amount']").text))
                            discount_percent = round(
                                (price - float(discounted_price)) / price, 2)
                        else:
                            price = float(re.sub(pattern, "", variant_hopp.select_one(
                                "span[class*='z-product-price__amount']").text))
                            discounted_price = None
                            discount_percent = None

                        variants.append(variant)
                        prices.append(price)
                        discounted_prices.append(discounted_price)
                        discount_percentages.append(discount_percent)
                        image_urls.append(image_variant)

                else:
                    variant = soup.select_one(
                        "div[data-zta*='ProductTitle__Subtitle']").text
                    discount_checker = soup.find('span', attrs={
                                                 'data-zta': 'SelectedArticleBox__TopSection'}).find('div', class_="z-product-price__note-wrap")

                    if discount_checker:
                        price = float(re.sub(rrb_pattern, "", soup.find('span', attrs={
                                      'data-zta': 'SelectedArticleBox__TopSection'}).find('div', class_="z-product-price__nowrap").get_text()))
                        discounted_price = float(re.sub(pattern, "", soup.find('span', attrs={
                                                 'data-zta': 'SelectedArticleBox__TopSection'}).find('span', class_="z-product-price__amount--reduced").get_text()))
                        discount_percent = round(
                            (price - float(discounted_price)) / price, 2)
                    else:
                        price = float(re.sub(pattern, "", soup.find('span', attrs={
                                      'data-zta': 'SelectedArticleBox__TopSection'}).find('span', class_="z-product-price__amount").get_text()))
                        discounted_price = None
                        discount_percent = None

                    variants.append(variant)
                    prices.append(price)
                    discounted_prices.append(discounted_price)
                    discount_percentages.append(discount_percent)
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
                df.insert(0, "description", description)
                df.insert(0, "rating", rating)
                df.insert(0, "name", product_title)
                df.insert(0, "shop", self.SHOP)

                return df
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
