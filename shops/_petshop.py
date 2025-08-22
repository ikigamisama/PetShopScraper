import math
import asyncio
import requests
import pandas as pd


from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class PetShopETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "PetShop"
        self.BASE_URL = "https://www.petshop.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = ''
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        category_link = f"{self.BASE_URL}{category}"
        urls = []
        soup = asyncio.run(self.scrape(
            category_link, 'div.facets-facet-browse-items'))

        n_products = int(soup.select_one(
            "h1[class='facets-facet-browse-title']")["data-quantity"])
        n_products_per_page = 50
        n_pages = math.ceil(n_products / n_products_per_page) + 1

        for p in range(1, n_pages):
            if p == 1:
                category_link_page = category_link
            else:
                category_link_page = f"{category_link}?page={p}"

            pagination_soup = asyncio.run(self.scrape(
                category_link_page, 'div.facets-facet-browse-items'))
            if pagination_soup:
                product_links_a = pagination_soup.select(
                    "a[class='facets-item-cell-grid-link-image']")
                product_links = [self.BASE_URL + plink["href"]
                                 for plink in product_links_a]
                urls.extend(product_links)

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)

        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_name = soup.find('h1', class_="product-details-full-content-header-title").find(
                string=True, recursive=False).get_text(strip=True)

            description_wrapper = soup.find(
                'div', id="product-details-information-tab-content-container-0")
            product_description = None

            if description_wrapper:
                product_description = description_wrapper.get_text(strip=True)

            product_url = url.replace(self.BASE_URL, "")
            product_rating = '0/5'

            rating_wrapper = soup.find(
                'div', class_="product-reviews-center-container-header")

            if rating_wrapper.find('h3', class_="product-reviews-center-container-header-number"):
                product_rating = soup.find(
                    'span', class_="global-views-star-rating-value").get_text(strip=True) + '/5'

            variant = None
            price = 0
            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            discount_price = None
            discount_percentage = None

            if " - " in product_name:
                variant = product_name.split(" - ")[1]
            elif "- " in product_name:
                variant = product_name.split("- ")[1]
            elif " -" in product_name:
                variant = product_name.split(" -")[1]

            variants.append(variant)
            image_urls.append(', '.join([img.find('img').get(
                'src') for img in soup.find('ul', class_="bxslider").find_all('li')]))

            get_price_details = requests.get(
                f"https://www.petshop.co.uk/api/cacheable/items?c=3934951&country=GB&currency=GBP&fieldset=details&include=facets&language=en&n=2&pricelevel=5&url={product_url.replace('/', '')}&use_pcv=T")
            if get_price_details.status_code == 200:
                product_info = get_price_details.json()['items'][0]
                if product_info.get('pricelevel2') is not None:
                    previous_price = product_info['pricelevel2']
                    current_price = product_info['pricelevel3']

                    if previous_price != current_price:
                        price = previous_price
                        discount_price = current_price
                        discount_percentage = (price - discount_price) / price
                else:
                    price = product_info['onlinecustomerprice_detail']['onlinecustomerprice']

            prices.append(price)
            discounted_prices.append(discount_price)
            discount_percentages.append(discount_percentage)

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
