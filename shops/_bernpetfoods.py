import re
import json
import requests
import asyncio
import math
import pandas as pd

from bs4 import BeautifulSoup
from functions.etl import PetProductsETL
from loguru import logger


class BernPetFoodsETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "BernPetFoods"
        self.BASE_URL = "https://www.bernpetfoods.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#primary'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        category_link = f"{self.BASE_URL}{category}"
        urls = []

        soup = asyncio.run(self.scrape(category_link, '#main-content'))
        if soup:
            n_products = 1
            if soup.find('p', class_="woocommerce-result-count").get_text(strip=True) != "Showing the single result":
                n_products = int(soup.find(
                    'p', class_="woocommerce-result-count").get_text().split(" of ")[1].replace(' results', ''))

            n_page = math.ceil(n_products / 18)

            for i in range(1, n_page + 1):
                page_url = f"{category_link}/page/{i}/"
                product_info_soup = asyncio.run(
                    self.scrape(page_url, '#main-content'))

                product_cards = product_info_soup.find_all(
                    "div", class_="ftc-product")
                product_links = [product_card.find(
                    "a")["href"] for product_card in product_cards]
                urls.extend(product_links)

            df = pd.DataFrame({"url": urls})
            df.insert(0, "shop", self.SHOP)
            return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_name = soup.find(
                'h1', class_="product_title").get_text(strip=True)
            product_description = soup.find(
                'div', class_="description_fullcontent").get_text(separator=' ', strip=True)
            product_url = url.replace(self.BASE_URL, "")

            product_id = re.search(
                r'postid-(\d+)', ' '.join(soup.body['class'])).group(0)

            rating_wrapper = requests.get(
                f"https://api.feefo.com/api/10/reviews/summary/product?since_period=ALL&parent_product_sku={product_id}&merchant_identifier=bern-pet-foods&origin=www.bernpetfoods.co.uk")
            rating = int(rating_wrapper.json()['rating']['rating'])
            product_rating = f'{rating}/5'

            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = [img.find("img")["src"] for img in soup.find_all(
                'div', class_="woocommerce-product-gallery__image")]

            if (soup.find('form', class_="variations_form")):
                for price_details in json.loads(soup.find('form', class_="variations_form").get('data-product_variations')):
                    variant = price_details.get('weight_html')
                    price = None
                    discounted_price = None
                    discount_percentage = None

                    if price_details.get('display_price') == price_details.get('display_regular_price'):
                        price = price_details.get('display_price')
                    else:
                        price = price_details.get('display_regular_price')
                        discounted_price = price_details.get(
                            'display_price')
                        discount_percentage = "{:.2f}".format(
                            (price - discounted_price) / price)

                    variants.append(variant)
                    prices.append(price)
                    discounted_prices.append(discounted_price)
                    discount_percentages.append(discount_percentage)

            else:
                variants.append(None)
                prices.append(
                    float(soup.find('p', class_="price").get_text().replace('£', '')))
                discounted_prices.append(None)
                discount_percentages.append(None)

            df = pd.DataFrame({
                "variant": variants,
                "price": prices,
                "discounted_price": discounted_prices,
                "discount_percentage": discount_percentages
            })
            df.insert(0, "url", product_url)
            df.insert(0, "description", product_description)
            df.insert(0, "rating", product_rating)
            df.insert(0, "name", product_name)
            df.insert(0, "shop", self.SHOP)
            df.insert(0, "image_urls", ", ".join(image_urls))

            return df
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
