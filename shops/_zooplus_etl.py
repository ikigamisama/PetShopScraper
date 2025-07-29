import asyncio
import requests
import re
import random
import json
import time
import pandas as pd

from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger
from fake_useragent import UserAgent


class ZooplusETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "Zooplus"
        self.BASE_URL = "https://www.zooplus.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#page-content'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 3

    def get_product_links(self, url, headers):
        try:
            # Parse request response
            response = requests.get(url=url, headers=headers)
            response.raise_for_status()

            logger.info(
                f"Successfully extracted data from {url} {response.status_code}"
            )
            sleep_time = random.uniform(
                self.MIN_SEC_SLEEP_PRODUCT_INFO, self.MAX_SEC_SLEEP_PRODUCT_INFO)
            time.sleep(sleep_time)
            logger.info(f"Sleeping for {sleep_time} seconds...")
            return response

        except Exception as e:
            logger.error(f"Error in parsing {url}: {e}")

    def extract(self, category):
        headers = {
            "Accept": 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'max-age=0',
            "User-Agent": UserAgent().random,
            'Referer': 'https://www.zooplus.co.uk',
            'Priority': "u=0, i",
            "Upgrade-Insecure-Requests": "1",
            "Connection": "keep-alive",
            "Sec-Ch-Ua": "\"Not(A:Brand\";v=\"99\", \"Opera GX\";v=\"118\", \"Chromium\";v=\"133\"",
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": "\"Windows\"",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1"
        }
        urls = []
        n_page_pagination = 1
        list_prod_api = self.get_product_links(
            f"https://www.zooplus.co.uk/api/discover/v1/products/list-faceted-partial?&path={category}&domain=zooplus.co.uk&language=en&page=1&size=24&ab=shop-10734_shop_product_catalog_api_enabled_targeted_delivery.enabled%2Bidpo-1141_article_based_product_cards_targeted_delivery.on%2Bidpo-1390_rebranding_foundation_targeted_delivery.on%2Bexplore-3092-price-redesign_targeted_delivery.on", headers=headers)
        if list_prod_api.status_code == 200:
            if list_prod_api.json()['pagination'] == None:
                for products in list_prod_api.json()['productList']['products']:
                    urls.append(self.BASE_URL + products["path"])

            else:
                n_page_pagination = int(list_prod_api.json()[
                                        'pagination']["count"])

        if n_page_pagination > 1:
            for i in range(1, n_page_pagination + 1):
                pagination_url = f"https://www.zooplus.co.uk/api/discover/v1/products/list-faceted-partial?&path={category}&domain=zooplus.co.uk&language=en&page={i}&size=24&ab=shop-10734_shop_product_catalog_api_enabled_targeted_delivery.enabled%2Bidpo-1141_article_based_product_cards_targeted_delivery.on%2Bidpo-1390_rebranding_foundation_targeted_delivery.on%2Bexplore-3092-price-redesign_targeted_delivery.on"

                pagination_product_api = self.get_product_links(
                    pagination_url, headers=headers)
                if pagination_product_api.status_code == 200:
                    for products in pagination_product_api.json()['productList']['products']:
                        urls.append(self.BASE_URL + products["path"])

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)
        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_data = json.loads(soup.select(
                "script[type*='application/ld+json']")[0].text)
            product_name = product_data['name']
            product_description = product_data['description']
            product_url = url.replace(self.BASE_URL, "")

            rating = '0/5'
            if "aggregateRating" in product_data.keys():
                rating = product_data["aggregateRating"]["ratingValue"]
                rating = f"{rating}/5"

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
            df.insert(0, "description", product_description)
            df.insert(0, "rating", rating)
            df.insert(0, "name", product_name)
            df.insert(0, "shop", self.SHOP)

            return df

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
