import asyncio
import re
import math
import pandas as pd
from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class VetUKETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "VetUK"
        self.BASE_URL = "https://www.vetuk.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#maincontent'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        urls = []
        soup = asyncio.run(self.scrape(
            category, '#productListing', min_sec=1, max_sec=2))

        heading = soup.find('h1', id="advSearchResultsDefaultHeading")

        if heading:
            match = re.search(r'\((\d+)\s+results\)', heading.get_text())

            n_product = int(match.group(1))
            n_pagination = math.ceil(n_product / 20)

            for n in range(1, n_pagination + 1):
                pagination_url = f"{category}&page={n}"
                pagination_soup = asyncio.run(self.scrape(
                    pagination_url, '.product-list-table', min_sec=1, max_sec=2))

                urls.extend([
                    link.find('a').get('href')
                    for link in pagination_soup.find_all('h3', class_="itemTitle")
                    if link.find('a') and link.find('a').get('href')
                ])
        else:
            n_product = int(soup.find('div', id="pagination").find_all(
                'strong')[2].get_text(strip=True))

            n_pagination = math.ceil(n_product / 20)

            for n in range(1, n_pagination + 1):
                pagination_url = f"{category}&page={n}"
                pagination_soup = asyncio.run(self.scrape(
                    pagination_url, '.product-list-table', min_sec=1, max_sec=2))

                urls.extend([
                    link.find('a').get('href')
                    for link in pagination_soup.find_all('h3', class_="itemTitle")
                    if link.find('a') and link.find('a').get('href')
                ])

        df = pd.DataFrame({"url": urls})
        df = df.drop_duplicates(subset=['url'], keep="first")
        df.insert(0, "shop", self.SHOP)
        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            variant_wrapper = soup.find_all('div', class_="priceOption")
            if soup.find(string="(Sold Out)"):
                variant_len = len(variant_wrapper)
                variant_sold_out = 0
                for variant in variant_wrapper:
                    if variant.find('span', string="(Sold Out)"):
                        variant_sold_out += 1

                if variant_sold_out == variant_len:
                    logger.info(f"Skipping {url} as it is sold out. ")
                    return None

            product_name = soup.find(
                'div', id="product-name").find('h1').get_text()
            product_url = url.replace(self.BASE_URL, "")
            product_description_wrapper = soup.find(
                'div', class_="products-description").find_all('p')
            descriptions = []
            for description_wrap in product_description_wrapper:
                if (description_wrap.find('span') == None or description_wrap.find('strong') == None):
                    descriptions.append(description_wrap.get_text())

            product_description = ' '.join(descriptions)

            rating_wrapper = soup.find('div', id='reviews')
            rating_count = int(rating_wrapper.find(
                'h3').get_text().replace('Reviews (', '').replace(')', ''))
            product_rating = ''
            if (rating_count > 0):
                product_rating = f'{rating_wrapper.find("span", class_="star-rating-widget").get("data-rating")}/5'
            else:
                product_rating = '0/5'

            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            for v in variant_wrapper:
                if "(Sold Out)" in v.find('span').get_text():
                    continue

                variant_name = ''
                price = 0

                if soup.find('select', id="attribute-selector"):
                    variant_name = v.find(
                        'p', class_='displayOptionName').get_text()
                else:
                    if "(" in product_name and ")" in product_name:
                        variant_name = product_name.split(
                            '(')[1].replace(')', '')
                    else:
                        variant_name = soup.find(
                            'p', class_="manufacturer-name").get_text().replace('Manufacturer: ', '')

                variants.append(variant_name)
                image_urls.append(
                    soup.find('img', class_="product-image-main").get('src'))

                if v.find('span', class_='retailPrice'):
                    if "Now: £" in v.find('span', class_='retailPrice').get_text():
                        price = v.find('span', class_='retailPrice').get_text().replace(
                            'Now: £', '')
                    else:
                        price = v.find(
                            'span', class_='retailPrice').get_text().replace('£', '')

                    price = float(price.replace(' (each)', ''))
                    prices.append(price)
                else:
                    prices.append(None)

                discount_percenrage = None
                if v.find('span', class_="discountSaving"):
                    if "Save: " in v.find('span', class_='discountSaving').get_text():
                        discount_percenrage = v.find('span', class_='discountSaving').get_text(
                        ).replace('Save: ', '').replace('%', '')
                    else:
                        discount_percenrage = v.find(
                            'span', class_='discountSaving').get_text().replace('£', '')

                discount_percentages.append(discount_percenrage)

                discount_price = None
                if v.find('span', class_="wasPrice"):
                    if "Was: £" in v.find('span', class_='wasPrice').get_text():
                        discount_price = v.find(
                            'span', class_='wasPrice').get_text().replace('Was: £', '')
                    else:
                        discount_price = v.find(
                            'span', class_='wasPrice').get_text().replace('£', '')

                    discounted_prices.append(float(discount_price))
                else:
                    discounted_prices.append(discount_price)

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
