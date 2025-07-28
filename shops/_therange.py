import os
import asyncio
import json
import random
import pandas as pd

from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger
from fake_useragent import UserAgent
from playwright.async_api import async_playwright
from datetime import datetime as dt
from patchright.async_api import async_playwright as patch_async_playwright


class TheRangeETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "TheRange"
        self.BASE_URL = "https://www.therange.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#product_reviews_section'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 3
        self.browser_type = "chromium"

    async def scrape_product_page(self, url, selector):
        logger.info(f"Navigating to: {url}")
        profile_dir = os.path.abspath("chrome_user_data")
        async with patch_async_playwright() as p:
            context = await p.chromium.launch_persistent_context(
                user_data_dir=profile_dir,
                channel="chrome",
                headless=False,
                no_viewport=True
            )
            page = context.pages[0]
            await page.goto(url)
            await page.wait_for_selector(selector)
            html = await page.content()
            # Interact with the page...
            await context.close()
        logger.success(f"Successfully extracted content from {url}")
        return BeautifulSoup(html, "html.parser")

    async def get_data_variant(self, url):
        browser = None
        data = None

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"]
                )

                context = await browser.new_context(
                    user_agent=UserAgent().random,
                    viewport={"width": random.randint(
                        1200, 1600), "height": random.randint(800, 1200)},
                    locale="en-US"
                )

                page = await context.new_page()

                # Capture first JSON response
                async def handle_response(response):
                    nonlocal data
                    try:
                        if "application/json" in response.headers.get("content-type", "") and response.status == 200:
                            json_data = await response.json()
                            logger.info(f"Captured JSON from: {response.url}")
                            data = json_data
                    except Exception as e:
                        logger.error(f"Failed to parse JSON: {e}")

                page.on("response", handle_response)

                await page.set_extra_http_headers({
                    "User-Agent": UserAgent().random,
                    "Accept-Language": "en-US,en;q=0.9",
                    "Origin": "https://www.therange.co.uk",
                    "Referer": url,
                })

                await page.goto(url, wait_until="networkidle")

                sleep_time = random.uniform(2, 5)
                logger.info(f"Sleeping for {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)

                return data

        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            if browser:
                await browser.close()

    def extract(self, category):
        category_link = f"https://www.therange.co.uk{category}"
        urls = []
        soup = asyncio.run(self.scrape(category_link,  '#root',
                           min_sec=self.MIN_SEC_SLEEP_PRODUCT_INFO, max_sec=self.MAX_SEC_SLEEP_PRODUCT_INFO, browser='chromium'))

        if not soup or not isinstance(soup, BeautifulSoup):
            logger.error(f"Failed to scrape category page: {category_link}")
            return pd.DataFrame(columns=["shop", "url"])

        root_div = soup.find('div', id="root")
        if not root_div:
            logger.error("Missing <div id='root'> in HTML")
            return pd.DataFrame(columns=["shop", "url"])

        try:
            n_product = root_div['data-total-results']
            category_id = root_div['data-page-id']
        except KeyError as e:
            logger.error(f"Missing expected attributes in root div: {e}")
            return pd.DataFrame(columns=["shop", "url"])

        product_data_list = asyncio.run(self.get_data_variant(
            f'https://search.therange.co.uk/api/productlist?categoryId={category_id}&sort=relevance&limit={n_product}&filters=%7B"in_stock_f"%3A%5B"true"%5D%7D'))

        urls = [self.BASE_URL + '/' + p['variantPath']
                for p in product_data_list.get('products', [])
                if p.get('variantPath')]

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)
        return df

    def get_product_infos(self):
        temp_table = f"stg_{self.SHOP.lower()}_temp_products"
        df_urls = self.extract_unscraped_data(temp_table)

        for i, row in df_urls.iterrows():
            pkey = row["id"]
            url = row["url"]

            now = dt.now().strftime("%Y-%m-%d %H:%M:%S")
            soup = asyncio.run(self.scrape_product_page(
                url, self.SELECTOR_SCRAPE_PRODUCT_INFO))

            df = self.transform(soup, url)

            if df is not None:
                self.load(df, temp_table)
                self.connection.update_url_scrape_status(
                    pkey, "DONE", 'urls', now)
            else:
                self.connection.update_url_scrape_status(
                    pkey, "FAILED", 'urls', now)

            logger.info(f"{i+1} out of {len(df_urls)} URL(s) Scraped")

        self.insert_scrape_in_database(temp_table)

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_name = soup.find('h1', id="product-dyn-title").get_text()
            product_description = soup.find(
                'p', id='product-dyn-desc').find(string=True)
            product_url = url.replace(self.BASE_URL, "")
            product_rating = "0/5"
            product_id = soup.find('input', id="product_id").get('value')
            clean_url = url.split('#')[0]

            product_rating_soup = asyncio.run(self.scrape_product_page(
                f'{clean_url}?action=loadreviews&pid={product_id}&page=1', 'body'))

            if product_rating_soup.find('div', id="review-product-summary"):
                product_rating = str(round((int(product_rating_soup.find('div', id="review-product-summary").findAll(
                    'div', class_="progress-bar")[0].get('aria-valuenow')) / 100) * 5, 2)) + '/5'

            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            product_details_soup = asyncio.run(
                self.scrape(f'{clean_url}?json', 'pre', wait_until='load', min_sec=self.MIN_SEC_SLEEP_PRODUCT_INFO, max_sec=self.MAX_SEC_SLEEP_PRODUCT_INFO, browser='chromium'))

            pre_tag = product_details_soup.find('pre')

            if pre_tag:
                product_details = json.loads(pre_tag.get_text())

            if len(product_details['variant_arr']) > 1:
                for var_details in product_details['variant_arr']:
                    if " - " in var_details['name']:
                        variants.append(var_details['name'].split(" - ")[1])

                    if var_details['price_was'] == None:
                        prices.append(var_details['price'] / 100)
                        discounted_prices.append(None)
                        discount_percentages.append(None)

                    else:
                        prices.append(var_details['price_was'] / 100)
                        discounted_prices.append(var_details['price'] / 100)
                        discount_percentages.append(
                            var_details['price_was_percent'] / 100)

                    image_urls.append(
                        soup.find('meta', attrs={'property': "og:image"}).get('content'))

            else:
                variants.append(None)
                image_urls.append(
                    soup.find('meta', attrs={'property': "og:image"}).get('content'))
                if product_details['variant_arr'][0]['price_was'] == None:
                    prices.append(
                        product_details['variant_arr'][0]['price'] / 100)
                    discounted_prices.append(None)
                    discount_percentages.append(None)
                else:
                    prices.append(
                        product_details['variant_arr'][0]['price_was'] / 100)
                    discounted_prices.append(
                        product_details['variant_arr'][0]['price'] / 100)
                    discount_percentages.append(
                        product_details['variant_arr'][0]['price_was_percent'] / 100)

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
