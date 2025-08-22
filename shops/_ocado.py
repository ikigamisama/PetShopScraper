import asyncio
import random
import time
import json
import pandas as pd

from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from playwright.async_api import async_playwright
from loguru import logger


class OcadoETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "Ocado"
        self.BASE_URL = "https://www.ocado.com"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#main'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    async def product_list_scrolling(self, url, selector, timeout: int = 60):
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
                    viewport={
                        "width": random.randint(1200, 1600),
                        "height": random.randint(800, 1200)
                    },
                    locale="en-US"
                )

                page = await context.new_page()
                await page.set_extra_http_headers({
                    "User-Agent": UserAgent().random,
                    "Accept-Language": "en-US,en;q=0.9",
                    "Origin": "https://www.ocado.com",
                    "Referer": url,
                })

                await page.goto(url, wait_until="domcontentloaded")
                await page.wait_for_selector(selector, timeout=30000)

                logger.info("Starting infinite scroll scrape...")

                scroll_delay = 2  # seconds
                start_time = time.time()
                last_height = await page.evaluate('() => document.body.scrollHeight')

                product_card_soups = []
                seen_cards = set()

                while True:
                    # Collect currently visible product cards
                    card_elements = await page.query_selector_all("div.sc-kdIgRK.cEKGlL")
                    for card in card_elements:
                        html = await card.inner_html()
                        if html not in seen_cards:
                            seen_cards.add(html)
                            soup_card = BeautifulSoup(html, "html.parser")
                            product_card_soups.append(
                                soup_card.find('a').get('href'))

                    # Scroll to the very bottom
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(scroll_delay)

                    # Check new page height
                    new_height = await page.evaluate('() => document.body.scrollHeight')

                    # Stop if no new content is loaded
                    if new_height == last_height:
                        logger.info("No more content loaded. Stopping scroll.")
                        break
                    last_height = new_height

                    # Stop if timeout exceeded
                    if time.time() - start_time > timeout:
                        logger.info(
                            f"Timeout of {timeout} seconds reached. Stopping scroll.")
                        break

                logger.info(
                    f"Scraping complete. Extracted {len(product_card_soups)} product cards.")

                return product_card_soups

        except Exception as e:
            logger.error(f"An error occurred: {e}")

        finally:
            if browser:
                await browser.close()

    def extract(self, category):
        category_link = f"{self.BASE_URL}{category}"

        product_list = asyncio.run(
            self.product_list_scrolling(f"{category_link}", '#product-page'))

        urls = [self.BASE_URL + product for product in product_list]

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)
        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            details = json.loads(soup.find(
                'script', attrs={'data-test': 'product-details-structured-data'}).get_text())

            product_name = details['name']
            product_description = details['description']
            product_url = url.replace(self.BASE_URL, "")
            product_rating = '0/5'

            if details.get('aggregateRating'):
                product_rating = str(
                    int(float(details['aggregateRating']['ratingValue']))) + '/5'

            price_details = json.loads(soup.find('script', attrs={
                                       'data-test': 'initial-state-script'}).get_text().replace('window.__INITIAL_STATE__=', ''))
            variant = details.get("brand", "") + " - " + \
                details.get("size", "")
            price = None
            discounted_price = None
            discount_percentage = None
            image_urls = "https://www.ocado.com" + \
                soup.find('meta', attrs={'property': "og:image"}).get(
                    'content')

            product_entities = price_details["data"]["products"]["productEntities"]
            price_data = product_entities[next(
                iter(product_entities))]['price']

            if price_data.get('original'):
                price = "{:.2f}".format(
                    float(price_data['original']['amount']))
                discounted_price = "{:.2f}".format(
                    float(price_data['current']['amount']))

                discount_percentage = round((float(price_data['original']['amount']) - float(
                    price_data['current']['amount'])) / float(price_data['original']['amount']), 2)

            else:
                price = "{:.2f}".format(float(price_data['current']['amount']))
                discounted_price = None
                discount_percentage = None

            df = pd.DataFrame([{
                "url": product_url,
                "description": product_description,
                "rating": product_rating,
                "name": product_name,
                "shop": "Ocado",
                "variant": variant,
                "price": price,
                "discounted_price": discounted_price,
                "discount_percentage": discount_percentage,
                "image_urls": image_urls
            }])

            return df

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
