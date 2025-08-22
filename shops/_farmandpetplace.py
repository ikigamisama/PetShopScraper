import math
import asyncio
import requests
import pandas as pd

from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class FarmAndPetPlaceETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "FarmAndPetPlace"
        self.BASE_URL = "https://www.farmandpetplace.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '.content-page'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2
        self.category_urls = []
        self.scrape_url_again = []
        self.scraped_urls = set()

    def _process_soup(self, soup, source_url):
        if not soup or isinstance(soup, bool):
            return

        if soup.find("div", class_="shop-filters-area"):
            if source_url and source_url not in self.category_urls:
                self.category_urls.append(source_url)

        if soup.find('div', class_="products-loop"):
            for div in soup.select("div.product-title a[href]"):
                full_url = self.BASE_URL + div["href"]
                if full_url not in self.scraped_urls:
                    self.scrape_url_again.append(full_url)
                    self.scraped_urls.add(full_url)

    async def rescrape_urls(self):
        while self.scrape_url_again:
            current_urls = self.scrape_url_again.copy()
            self.scrape_url_again.clear()

            for url in current_urls:
                soup = await self.scrape(
                    url, '.main-products-loop', wait_until='load',
                    min_sec=1, max_sec=3
                )
                self._process_soup(soup, url)

    def extract(self, category):
        urls = []

        self.category_urls = []
        self.scrape_url_again = []
        self.scraped_urls = set()

        soup = asyncio.run(self.scrape(
            category, '.main-products-loop', wait_until='load',
            min_sec=1, max_sec=2
        ))

        if not soup or isinstance(soup, bool):
            return pd.DataFrame({})

        self._process_soup(soup, category)

        if self.scrape_url_again:
            asyncio.run(self.rescrape_urls())

        for url_category in list(set(self.category_urls)):
            soup = asyncio.run(self.scrape(
                url_category, 'body.product-cats', min_sec=1, max_sec=3, wait_until="domcontentloaded"))

            if not soup or isinstance(soup, bool):
                print(f"[ERROR] Failed to scrape category page: {category}")
                return pd.DataFrame(columns=["shop", "url"])

            result_count = soup.find('p', class_="woocommerce-result-count")
            if result_count:
                words = result_count.get_text().split()
                n_product = next((int(w) for w in words if w.isdigit()), 0)
            else:
                n_product = 0

            n_pagination = math.ceil(n_product / 24)

            if n_pagination == 1:
                shop_area = soup.find('div', class_="shop-filters-area")
                if shop_area:
                    urls.extend([
                        self.BASE_URL + a_tag.get('href')
                        for product in shop_area.find_all('div', class_="product")
                        if (a_tag := product.find('a')) and a_tag.get('href')
                    ])
            else:
                for i in range(1, n_pagination + 1):
                    base = url_category.split("page-")[0]
                    new_url = f"{base}page-{i}.html"

                    soup_page = asyncio.run(
                        self.scrape(new_url, 'div.shop-filters-area',
                                    min_sec=1, max_sec=3)
                    )

                    if not soup_page:
                        continue

                    shop_area = soup_page.find(
                        'div', class_="shop-filters-area")
                    if shop_area:
                        urls.extend([
                            self.BASE_URL + a_tag.get('href')
                            for product in shop_area.find_all('div', class_="product")
                            if (a_tag := product.find('a')) and a_tag.get('href')
                        ])

        df = pd.DataFrame({"url": urls})
        df = df.drop_duplicates(subset=['url'])
        df.insert(0, "shop", self.SHOP)
        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_name = soup.find('h1', attrs={'itemprop': 'name'})
            if product_name:
                product_name = product_name.get_text()
            else:
                return pd.DataFrame({})

            product_description = None

            if soup.find('div', class_="short-description"):
                product_description = soup.find(
                    'div', class_="short-description").get_text(strip=True)

            product_url = url.replace(self.BASE_URL, "")
            product_id = soup.find(
                'div', class_="ruk_rating_snippet").get('data-sku')

            try:
                rating_wrapper = requests.get(
                    f"https://api.feefo.com/api/10/reviews/summary/product?since_period=ALL&parent_product_sku={product_id}&merchant_identifier=farm-pet-place&origin=www.farmandpetplace.co.uk")

                if rating_wrapper.status_code == 200 and rating_wrapper.json().get('rating', {}).get('rating'):
                    rating = float(rating_wrapper.json()['rating']['rating'])
                    product_rating = f'{rating}/5'
                else:
                    product_rating = '0/5'

            except (requests.RequestException, KeyError, ValueError, TypeError):
                product_rating = '0/5'

            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            if soup.find('select', id="attribute"):
                variants.append(soup.find('select', id="attribute").find_all(
                    'option')[0].get('value'))
                image_urls.append(
                    soup.find('img', class_="attachment-shop_single").get('src'))
                if soup.find('div', class_="price").find('span', class_="rrp"):
                    price = float(soup.find('div', class_="price").find(
                        'span', class_="rrp").find('strong').get_text().replace('£', ''))
                    discounted_price = float(soup.find('div', class_="price").find(
                        'span', class_="current").find('strong').get_text().replace('£', ''))
                    discount_percentage = "{:.2f}".format(
                        (price - discounted_price) / price)

                    prices.append(price)
                    discounted_prices.append(discounted_price)
                    discount_percentages.append(discount_percentage)

                else:
                    prices.append(float(soup.find('div', class_="price").find(
                        'span', class_="current").find('strong').get_text().replace('£', '')))
                    discounted_prices.append(None)
                    discount_percentages.append(None)

            else:
                variants.append(None)
                image_urls.append(
                    soup.find('img', class_="attachment-shop_single").get('src'))
                if soup.find('div', class_="price").find('span', class_="rrp"):
                    price = float(soup.find('div', class_="price").find(
                        'span', class_="rrp").find('strong').get_text().replace('£', ''))
                    discounted_price = float(soup.find('div', class_="price").find(
                        'span', class_="current").find('strong').get_text().replace('£', ''))
                    discount_percentage = "{:.2f}".format(
                        (price - discounted_price) / price)

                    prices.append(price)
                    discounted_prices.append(discounted_price)
                    discount_percentages.append(discount_percentage)

                else:
                    prices.append(float(soup.find('div', class_="price").find(
                        'span', class_="current").find('strong').get_text().replace('£', '')))
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
