import asyncio
import random
import json
import requests
import pandas as pd
from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from playwright.async_api import async_playwright
from loguru import logger


class FishKeeperETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "FishKeeper"
        self.BASE_URL = "https://www.fishkeeper.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#maincontent'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    async def product_list_scroll(self, url, selector):
        soup = None
        browser = None
        try:
            async with async_playwright() as p:
                browser_args = {
                    "headless": True,
                    "args": ["--disable-blink-features=AutomationControlled"]
                }

                browser = await p.chromium.launch(**browser_args)
                context = await browser.new_context(
                    user_agent=UserAgent().random,
                    viewport={"width": random.randint(
                        1200, 1600), "height": random.randint(800, 1200)},
                    locale="en-US"
                )

                page = await context.new_page()
                await page.set_extra_http_headers({
                    "User-Agent": UserAgent().random,
                    "Accept-Language": "en-US,en;q=0.9",
                    "Origin": "https://www.fishkeeper.co.uk",
                    "Referer": url,
                })

                await page.goto(url, wait_until="domcontentloaded")
                await page.wait_for_selector(selector, timeout=30000)

                logger.info(
                    "Starting to scrape the product list (Infinite scroll scrape)...")

                while True:
                    try:
                        await page.wait_for_selector('.ais-InfiniteHits-loadMore', timeout=3000)
                        logger.info("Expanding Product List")
                        await page.click('.ais-InfiniteHits-loadMore')
                        await asyncio.sleep(1)
                    except Exception:
                        logger.info(
                            "Cannot Expand Product List Anymore Scraping Cmplete")
                        break

                rendered_html = await page.content()
                logger.info(
                    f"Successfully extracted data from {url}"
                )
                sleep_time = random.uniform(
                    2, 5)
                logger.info(f"Sleeping for {sleep_time} seconds...")
                soup = BeautifulSoup(rendered_html, "html.parser")
                return soup.find('ol', class_="ais-InfiniteHits-list")

        except Exception as e:
            logger.error(f"An error occurred: {e}")

        finally:
            if browser:
                await browser.close()

    def extract(self, category):
        url = self.BASE_URL + category

        soup_pagination = asyncio.run(
            self.product_list_scroll(url, '.ais-InfiniteHits-list'))
        urls = [product.find('a').get('href') for product in soup_pagination.find_all(
            'li', class_="ais-InfiniteHits-item")]

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)
        return df

    def transform(self,  soup: BeautifulSoup, url: str):
        try:
            data = json.loads(soup.select_one(
                "script[type*='application/ld+json']").text)
            product_title = data["name"]

            rating = 0
            sku = data["mpn"]
            rating_wrapper = requests.get(
                f"https://api.feefo.com/api/10/products/ratings?merchant_identifier=maidenhead-aquatics&review_count=true&product_sku={sku}")
            if rating_wrapper.status_code == 200:
                json_data = rating_wrapper.json()
                products = json_data.get('products', [])

                if products and 'rating' in products[0]:
                    rating = float(products[0]['rating'])

            description = data["description"]
            product_url = url.replace(self.BASE_URL, "")

            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            data_offers = data["offers"]

            if "offers" in data_offers.keys():
                for variant in data_offers["offers"]:
                    variants.append(variant['name'])
                    prices.append(variant['price'])
                    discounted_prices.append(None)
                    discount_percentages.append(None)
                    image_urls.append(variant['image'])

            else:
                variants.append(None)
                prices.append(data["offers"]['price'])
                discounted_prices.append(None)
                discount_percentages.append(None)
                image_urls.append(data['image'])

            df = pd.DataFrame({"variant": variants,
                               "price": prices,
                               "discounted_price": discounted_prices,
                               "discount_percentage": discount_percentages,
                               "image_urls": image_urls})

            df.insert(0, "url", product_url)
            df.insert(0, "description", description)
            df.insert(0, "rating", rating)
            df.insert(0, "name", product_title)
            df.insert(0, "shop", self.SHOP)

            return df

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
