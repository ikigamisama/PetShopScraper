import json
import asyncio
import pandas as pd
from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class HealthyPetStoreETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "HealthyPetStore"
        self.BASE_URL = "https://healthypetstore.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#wrapper'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 3
        self.wait_until = "domcontentloaded"
        self.browser_type = 'firefox'

    def extract(self, category):
        url = self.BASE_URL + category
        soup = asyncio.run(self.scrape(
            f"{url}?showall=1", '.thb-shop-content'))

        if not soup:
            logger.error(f"[WARN] No content found for category: {category}")
            return pd.DataFrame({})

        try:
            product_list = soup.find('ul', class_="products")
            urls = [product.find('a').get(
                'href') for product in product_list.find_all('li', class_="product")]
        except AttributeError:
            logger.error(f"[ERROR] Unexpected page structure for: {url}")
            return pd.DataFrame({})

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)

        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            if soup.find('h1', class_="product_title"):
                product_name = soup.find(
                    'h1', class_="product_title").get_text()
            else:
                return pd.DataFrame({})

            product_description = None

            if soup.find('div', class_="woocommerce-product-details__short-description"):
                product_description = soup.find(
                    'div', class_="woocommerce-product-details__short-description").get_text(strip=True)

            product_url = url.replace(self.BASE_URL, "")
            product_rating = '0/5'

            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            if soup.find('form', class_="variations_form"):
                for price_data in json.loads(soup.find('form', class_="variations_form").get('data-product_variations')):
                    variants.append(price_data['attributes'].get(
                        'attribute_pa_variations-sizes') or price_data['attributes'].get('attribute_pa_size'))
                    image_urls.append(soup.find('meta', {'property': 'og:image'}).get(
                        'content') if soup.find('meta', {'property': 'og:image'}) else None)

                    if price_data.get('display_price') != price_data.get('display_regular_price'):
                        price = float(price_data.get('display_regular_price'))
                        discounted_price = float(
                            price_data.get('display_price'))
                        discount_percentage = "{:.2f}".format(
                            (price - discounted_price) / price)

                        prices.append(price)
                        discounted_prices.append(discounted_price)
                        discount_percentages.append(discount_percentage)
                    else:
                        prices.append(float(price_data.get('display_price')))
                        discounted_prices.append(None)
                        discount_percentages.append(None)

            else:
                variants.append(None)
                image_urls.append(soup.find('meta', {'property': 'og:image'}).get(
                    'content') if soup.find('meta', {'property': 'og:image'}) else None)

                if soup.find('p', class_="price").find('del'):
                    price = float(soup.find('p', class_="price").find(
                        'del').find('bdi').get_text().replace('£', ''))
                    discounted_price = float(soup.find('p', class_="price").find(
                        'ins').find('bdi').get_text().replace('£', ''))
                    discount_percentage = "{:.2f}".format(
                        (price - discounted_price) / price)

                    prices.append(price)
                    discounted_prices.append(discounted_price)
                    discount_percentages.append(discount_percentage)
                else:
                    prices.append(float(soup.find('p', class_="price").find(
                        'bdi').get_text().replace('£', '')))
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

    def get_links(self, category: str) -> pd.DataFrame:
        if category not in self.CATEGORIES:
            raise ValueError(
                f"Invalid category. Value must be in {self.CATEGORIES}")

        url = self.BASE_URL+category
        soup = self.extract_from_url("GET", f"{url}?showall=1")

        urls = [product.find('a').get('href') for product in soup.find(
            'ul', class_="products").find_all('li', class_="product")]
        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)

        return df
