import asyncio
import math
import requests
import pandas as pd
from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class VetShopETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "VetShop"
        self.BASE_URL = "https://www.vetshop.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#product-detail'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        category_link = f"{self.BASE_URL}{category}"
        urls = []
        soup = asyncio.run(self.scrape(
            category_link, 'div.facets-facet-browse-results'))
        n_products = int(soup.select_one(
            "h1[class='facets-facet-browse-title']")["data-quantity"])
        n_products_per_page = 24
        n_pages = math.ceil(n_products / n_products_per_page) + 1

        for p in range(1, n_pages):
            if p == 1:
                category_link_page = category_link
            else:
                category_link_page = f"{category_link}?page={p}"

            soup_page = asyncio.run(self.scrape(
                category_link_page, 'div.facets-facet-browse-results'))
            if soup_page:
                product_links_a = soup_page.select(
                    "a[class='facets-item-cell-grid-link-image']")
                product_links = [self.BASE_URL + plink["href"]
                                 for plink in product_links_a]
                urls.extend(product_links)

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)

        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_name = soup.find(
                'h1', class_="item-details-content-header-title").find(string=True, recursive=False).get_text()

            description_wrapper = soup.find(
                'div', id="item-details-content-container-0")
            product_description = None

            if description_wrapper:
                product_description = description_wrapper.get_text(strip=True)

            product_url = url.replace(self.BASE_URL, "")
            product_rating = '0/5'
            rating_wrapper = soup.find(
                'div', class_="product-reviews-center-container-header")

            if rating_wrapper.find('h3', class_="product-reviews-center-container-header-number"):
                product_rating = rating_wrapper.find(
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

            if "-" in product_name:
                variant = product_name.split("-")[1]

            product_img = soup.find('meta', attrs={'name': "og:image"}).get(
                'content').replace('%2520', '%20')
            variants.append(variant)
            image_urls.append(product_img)

            was_price = soup.find(
                'div', class_="item-views-blb-price-options-compare-price")

            if was_price:
                price = float(was_price.find(
                    'span').get_text().replace('£', ''))
                discount_price = float(soup.find_all(
                    'p', class_="item-views-blb-price-option-price")[1].get_text().replace('£', ''))
                discount_percentage = (price - discount_price) / price

                prices.append(price)
                discounted_prices.append(discount_price)
                discount_percentages.append(discount_percentage)
            else:
                if soup.find('div', class_="item-views-blb-price-options-compare-price"):
                    price = float(soup.find_all(
                        'p', class_="item-views-blb-price-option-price")[1].get_text().replace('£', ''))
                else:
                    get_price_details = requests.get(
                        f"https://www.vetshop.co.uk/api/items?c=3934951&country=GB&currency=GBP&fields=pricelevel4%2Cpricelevel4_formatted&fieldset=details&include=facets&language=en&n=3&pricelevel=4&url={product_url.replace('/', '')}")
                    if get_price_details.status_code == 200:
                        product_info = get_price_details.json()['items'][0]

                        variants.clear()
                        prices.clear()
                        discounted_prices.clear()
                        discount_percentages.clear()

                        for product in product_info["matrixchilditems_detail"]:
                            if product.get('pricelevel4') is not None:
                                variants.append(product.get('custitem_bb1_size') or product.get(
                                    'custitem_bb1_packsize'))
                                prices.append(product['pricelevel4'])
                                discounted_prices.append(None)
                                discount_percentages.append(None)
                                image_urls.append(product_img)

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
