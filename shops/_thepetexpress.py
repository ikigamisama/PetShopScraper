import asyncio
import re
import pandas as pd
from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class ThePetExpressETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "ThePetExpress"
        self.BASE_URL = "https://www.thepetexpress.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '.product_page'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        url = self.BASE_URL + category
        soup = asyncio.run(self.scrape(url, '.category-view'))

        if not soup or isinstance(soup, bool):
            return pd.DataFrame({})

        count_div = soup.find('div', class_="pagination--count")
        if not count_div:
            return pd.DataFrame({})

        text = count_div.get_text(strip=True)
        match = re.search(r'[\d,]+', text)
        product_count = int(match.group().replace(",", "")) if match else 0

        real_soup = asyncio.run(self.scrape(
            f"{url}?limit={product_count}", '.category-view'))
        if not real_soup or isinstance(real_soup, bool):
            return pd.DataFrame({})

        urls = []
        for links in real_soup.find_all('div', class_="category-page"):
            a_tag = links.find('a')
            if a_tag and a_tag.get('href'):
                urls.append(self.BASE_URL + a_tag.get('href'))

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)
        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_name = soup.find(
                'div', class_="page-header").find('h1').get_text()
            product_description = None

            if soup.find('div', id="reviews"):
                product_rating = soup.find('div', id="reviews").find(
                    'span', class_="average_stars").get_text(strip=True)

            product_url = url.replace(self.BASE_URL, "")
            product_rating = '0/5'

            if soup.find('div', id="reviews"):
                product_rating = soup.find('div', id="reviews").find(
                    'span', class_="average_stars").get_text(strip=True)

            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            if soup.find('div', class_="in_page_options_option"):

                for variant in soup.find('div', class_="in_page_options_option").find_all('div', class_="sub-options"):
                    variants.append(variant.find(
                        'div', class_="inpage_option_title").get_text())
                    image_urls.append(
                        soup.find('meta', attrs={'property': "og:image"}).get('content'))

                    if variant.find('span', class_="inpage_option_rrp"):
                        price = float(variant.find(
                            'span', class_="inpage_option_rrp").get_text().replace('RRP: £', ''))
                        discount_price = float(variant.find(
                            'div', class_="ajax-price").get_text().replace('£', ''))
                        discount_percentage = round(
                            (price - float(discount_price)) / price, 2)

                        prices.append(price)
                        discounted_prices.append(discount_price)
                        discount_percentages.append(discount_percentage)

                    else:
                        prices.append(
                            float(variant.find('div', class_="ajax-price").get_text().replace('£', '')))
                        discounted_prices.append(None)
                        discount_percentages.append(None)

            else:
                variants.append(None)
                image_urls.append(
                    soup.find('meta', attrs={'property': "og:image"}).get('content'))

                is_price_same = soup.find('span', class_="ajax-price-vat").get_text().replace(
                    '£', '') == soup.find('span', class_="ajax-rrp").get_text().replace('£', '')

                if is_price_same or soup.find('span', class_="ajax-rrp").get_text() == "£0.00":
                    prices.append(
                        float(soup.find('span', class_="ajax-price-vat").get_text().replace('£', '')))
                    discounted_prices.append(None)
                    discount_percentages.append(None)

                else:
                    price = float(
                        soup.find('span', class_="ajax-rrp").get_text().replace('£', ''))
                    discount_price = float(
                        soup.find('span', class_="ajax-price-vat").get_text().replace('£', ''))
                    discount_percentage = round(
                        (price - float(discount_price)) / price, 2)

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
