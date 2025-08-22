import math
import asyncio
import requests
import pandas as pd
from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class PetsCornerETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "PetsCorner"
        self.BASE_URL = "https://www.petscorner.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#content'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        category_link = f"{self.BASE_URL}{category}"
        urls = []

        soup = asyncio.run(self.scrape(category_link, '.ProductListing'))
        if not soup:
            return pd.DataFrame({})

        try:
            total_text = soup.find('span', class_="total")
            if not total_text:
                return pd.DataFrame({})

            n_products = int(total_text.get_text().replace(' products', ''))
            n_pages = math.ceil(n_products / 48)
        except Exception as e:
            return pd.DataFrame({})

        for p in range(1, n_pages + 1):
            page_url = f'{category_link}?listing_page={p}'
            soup_pagination = asyncio.run(
                self.scrape(page_url, '.ProductListing'))

            if not soup_pagination:
                return pd.DataFrame({})

            try:
                product_divs = soup_pagination.find_all(
                    'div', class_="product-listing-column")
                for product in product_divs:
                    a_tag = product.find('a')
                    if a_tag and a_tag.get('href'):
                        urls.append(self.BASE_URL + a_tag.get('href'))
            except Exception as e:
                continue

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)
        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_name = soup.find('h1', class_="product-name").get_text()
            product_description = None

            if soup.find('div', id="ctl00_Content_zneContent6_ctl05_ctl02"):
                product_description = soup.find(
                    'div', id="ctl00_Content_zneContent6_ctl05_ctl02").get_text()

            product_url = url.replace(self.BASE_URL, "")
            product_rating = '0/5'
            product_id = soup.find_all(
                'div', class_="notify-stock")[-1].get('data-productid')
            sku = None
            sku_tag = soup.find('div', id="feefo-product-review-widgetId")
            if sku_tag.get('data-parent-product-sku'):
                sku = f"parent_product_sku={sku_tag.get('data-parent-product-sku')}"
            else:
                sku = f"product_sku={sku_tag.get('data-product-sku')}"

            rating_wrapper = requests.get(
                f"https://api.feefo.com/api/10/importedreviews/summary/product?since_period=ALL&{sku}&merchant_identifier=pets-corner&origin=www.petscorner.co.uk")
            if rating_wrapper.status_code == 200:
                product_rating = str(rating_wrapper.json()[
                                     'rating']['rating']) + '/5'

            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            if soup.find('div', class_="hidden-select"):
                for variant in soup.find('div', class_="fake-select").find_all('div', class_="text"):
                    variants.append(variant.get_text(strip=True))

                for price_list in soup.find('div', class_="hidden-select").find_all('input'):
                    if price_list.get('data-was-price') == '0.00':
                        prices.append(
                            float(price_list.get('data-product-price')))
                        discounted_prices.append(None)
                        discount_percentages.append(None)
                    else:
                        prices.append(float(price_list.get('data-was-price')))
                        discounted_prices.append(
                            float(price_list.get('data-product-price')))
                        discount_percentage = (float(price_list.get('data-was-price')) - float(
                            price_list.get('data-product-price'))) / float(price_list.get('data-was-price'))
                        discount_percentages.append(
                            "{:.2f}".format(round(discount_percentage, 2)))

                    image_urls.append(
                        soup.find('meta', attrs={'property': "og:image"}).get('content'))

            else:
                price_template = soup.find_all(
                    'span', attrs={'class': ['item-price', 'order-section']})[-1]

                if price_template.find('span', class_="was-price"):
                    variants.append(None)
                    prices.append(price_template.find(
                        'span', class_="was-price").get_text())
                    discounted_prices.append(
                        float(price_template.find('span', class_="price").get_text()))
                    discount_percentages.append(None)
                else:
                    variants.append(None)
                    prices.append(float(price_template.find(
                        'span', class_="price").get_text()))
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
