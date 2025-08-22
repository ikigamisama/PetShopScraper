import re
import pandas as pd
import asyncio

from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class AsdaETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "ASDAGroceries"
        self.BASE_URL = "https://groceries.asda.com"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = 'main.layout__main'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        category_link = f"{self.BASE_URL}{category}"
        urls = []

        soup = asyncio.run(self.scrape(category_link, '.layout__main'))

        try:

            if soup.find('div', class_="co-pagination"):
                n_pages = int(
                    soup.find('div', class_="co-pagination__max-page").text)

                for p in range(1, n_pages):
                    soup_page_pagination = asyncio.run(
                        self.scrape(f"{category_link}?page={p}", '#main-content'))
                    for product_container in soup_page_pagination.find_all('ul', class_="co-product-list__main-cntr"):
                        for product_list in product_container.find_all('li'):
                            if product_list.find('a'):
                                urls.append(self.BASE_URL +
                                            product_list.find('a').get('href'))

            else:
                for product_container in soup.find_all('ul', class_="co-product-list__main-cntr"):
                    for product_list in product_container.find_all('li'):
                        if product_list.find('a'):
                            urls.append(self.BASE_URL +
                                        product_list.find('a').get('href'))

            df = pd.DataFrame({"url": urls})
            df.insert(0, "shop", self.SHOP)
            return df

        except Exception as e:
            logger.error(f"Error scraping {category_link}: {e}")
            return pd.DataFrame({})

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            if soup.find('main', class_='product-detail-page'):
                product_name = soup.find(
                    'h1', class_="pdp-main-details__title").get_text()

                description_wrapper = soup.find(
                    'div', class_="pdp-description-reviews__product-details-cntr")
                product_description = None

                if description_wrapper:
                    product_description = description_wrapper.get_text()

                product_url = url.replace(self.BASE_URL, "")
                product_rating = '0/5'
                product_wrapper = soup.find(
                    'div', class_="pdp-main-details__rating")

                if product_wrapper:
                    product_rating = product_wrapper.get(
                        'aria-label').split(" ")[0] + '/5'

                variant = None
                price = float(soup.find('div', class_="pdp-main-details__price-container").find('strong', {'class': [
                    'co-product__price', 'pdp-main-details__price']}).find(string=True, recursive=False).strip().replace('£', ''))
                variants = []
                prices = []
                discounted_prices = []
                discount_percentages = []
                image_urls = []

                if soup.find('div', class_="pdp-main-details__weight"):
                    variants.append(
                        soup.find('div', class_="pdp-main-details__weight").get_text())
                else:
                    variants.append(variant)

                image_urls.append(
                    soup.find('meta', attrs={'property': "og:image"}).get('content'))

                price = float(re.search(r"(\d+\.\d+)", soup.find('strong',
                                                                 class_="co-product__price pdp-main-details__price").text).group(1))
                was_price_tag = soup.find(
                    'span', class_="co-product__was-price pdp-main-details__was-price")

                if was_price_tag:
                    real_price_text = was_price_tag.text
                    real_price_match = re.search(
                        r"(\d+\.\d+)", real_price_text)

                    if real_price_match:
                        real_price = float(real_price_match.group(1))

                        prices.append(real_price)
                        discounted_prices.append(price)
                        discount_percentages.append(
                            round((real_price - price) / real_price, 2))

                else:
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
            else:
                return pd.DataFrame({})
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
