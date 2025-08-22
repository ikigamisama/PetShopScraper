import json
import math
import asyncio
import pandas as pd

from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class PetsAtHomeETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "PetsAtHome"
        self.BASE_URL = "https://www.petsathome.com"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = ''
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        urls = []
        url = self.BASE_URL + f"/product/listing/{category}"

        soup = asyncio.run(self.scrape(
            url, '.search-results_grid__rmdgH', wait_until='load'))

        if not soup:
            logger.error(f"[ERROR] Initial scrape failed for URL: {url}")
            return pd.DataFrame(columns=["shop", "url"])

        try:
            nav_tag = soup.find('nav', class_="results-per-page_root__aknxt")
            if not nav_tag:
                logger.warning(
                    f"[WARNING] Could not find product count container in {url}")
                return pd.DataFrame(columns=["shop", "url"])

            n_product = int(nav_tag.find('strong').get_text(strip=True))
            n_pagination = math.ceil(n_product / 40)
        except Exception as e:
            logger.error(
                f"[ERROR] Failed to extract product count from {url}: {e}")
            return pd.DataFrame(columns=["shop", "url"])

        for n in range(1, n_pagination + 1):
            pagination_url = url + f'?page={n}'
            page_soup = asyncio.run(self.scrape(
                pagination_url, '.search-results_grid__rmdgH', wait_until='load'))

            if not page_soup:
                logger.warning(
                    f"[WARNING] Skipping page {n} due to empty or invalid soup.")
                continue

            try:
                items = page_soup.find_all(
                    'li', class_="results-grid_item__BuYWN")
                urls.extend([
                    self.BASE_URL + link.find('a').get('href')
                    for link in items
                    if link.find('a') and link.find('a').get('href')
                ])
            except Exception as e:
                logger.error(
                    f"[ERROR] Failed to extract URLs from {pagination_url}: {e}")
                continue

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)

        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            # Get the data from encoded JSON
            product_data = soup.select_one("[id='__NEXT_DATA__']")
            product_data_dict = json.loads(product_data.text)

            # Get base details
            product_title = product_data_dict["props"]["pageProps"]["baseProduct"]["name"]
            rating = product_data_dict["props"]["pageProps"]["productRating"]
            rating = product_data_dict["props"]["pageProps"]["productRating"]
            if rating:
                rating = "{} out of 5".format(rating["averageRating"])
            else:
                rating = None
            description = product_data_dict["props"]["pageProps"]["baseProduct"]["description"]
            product_url = url.replace("https://www.petsathome.com", "")

            # Placeholder for variant details
            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            # Iterate through all product variants
            for variant in product_data_dict["props"]["pageProps"]["baseProduct"]["products"]:
                variants.append(variant["label"])

                price = variant["price"]["base"]
                discounted_price = variant["price"]["promotionBase"]

                prices.append(price)
                discounted_prices.append(discounted_price)

                if discounted_price:
                    discount_percentage = (price - discounted_price) / price
                else:
                    discount_percentage = None

                discount_percentages.append(discount_percentage)
                image_urls.append(', '.join(variant['imageUrls']))

            # Compile the data acquired into dataframe
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
            df.insert(0, "description", description)
            df.insert(0, "rating", rating)
            df.insert(0, "name", product_title)
            df.insert(0, "shop", self.SHOP)

            return df

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
