import asyncio
import pandas as pd
import json
import math
import re
import random
from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger

from fake_useragent import UserAgent
from playwright.async_api import async_playwright


class JollyesETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "Jollyes"
        self.BASE_URL = "https://www.jollyes.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#viewport'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 4
        self.browser_type = 'chromium'

    async def product_list_scrolling(self, url, selector, click_times):
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
                    "Origin": "https://www.jollyes.co.uk",
                    "Referer": url,
                })

                await page.goto(url, wait_until="load")
                await page.wait_for_selector(selector, timeout=60000)

                logger.info(
                    "Starting to click 'Load More' button if available...")

                for i in range(click_times):
                    try:
                        load_more_btn = await page.query_selector("div.progress-row a")
                        if load_more_btn and await load_more_btn.is_visible():
                            current_products = await page.query_selector_all("a.product-link")
                            count_before = len(current_products)

                            await load_more_btn.click()
                            logger.info(
                                f"Clicked 'Load More' button ({i + 1}/{click_times})")

                            await page.wait_for_function(
                                f'document.querySelectorAll("a.product-link").length > {count_before}',
                                timeout=120000
                            )

                        else:
                            logger.warning(
                                "Load More button not found or not visible. Stopping clicks early.")
                            break
                    except Exception as e:
                        logger.warning(f"Error during click {i + 1}: {e}")
                        break

                logger.info("Scraping complete. Extracting content...")

                rendered_html = await page.content()
                logger.info(
                    f"Successfully extracted data from {url}"
                )
                sleep_time = random.uniform(
                    3, 5)
                logger.info(f"Sleeping for {sleep_time} seconds...")
                soup = BeautifulSoup(rendered_html, "html.parser")
                return soup.find_all('a', class_="product-link")

        except Exception as e:
            logger.error(f"An error occurred: {e}")

        finally:
            if browser:
                await browser.close()

    def extract(self, category):
        category_link = f"{self.BASE_URL}/{category}.html"
        soup = asyncio.run(self.scrape(
            category_link, '#category', wait_until="networkidle"))

        subcategory_links = [link["href"] for ul in soup.select(
            "ul.second-category") for link in ul.select("a")]

        urls = []

        for subcategory in subcategory_links:
            url = self.BASE_URL + subcategory

            category_soup = asyncio.run(self.scrape(
                url, '.product-list', wait_until="networkidle", min_sec=3, max_sec=5))

            if not category_soup:
                logger.error(f"[ERROR] Failed to fetch or parse: {url}")
                continue

            sorting_row = category_soup.find('div', class_="sorting-row")
            if sorting_row:
                p_tag = sorting_row.find('p')
                if p_tag:
                    n_products_results = p_tag.get_text(strip=True)
                    n_products = int(re.findall(r'\d+', n_products_results)[0])
                else:
                    logger.warning(
                        f"[WARN] No <p> tag found in sorting row for {url}")
                    continue
            else:
                logger.warning(f"[WARN] No 'sorting-row' found for {url}")
                continue

            n_pagination = math.ceil(n_products / 40)

            product_links = asyncio.run(self.product_list_scrolling(
                url, '.product-list', n_pagination))

            if product_links:
                urls.extend([self.BASE_URL + links.get('href')
                            for links in product_links])

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)
        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            data = json.loads(soup.select_one(
                "section[class*='lazy-review-section']").select_one("script[type*='application']").text)
            product_title = data["name"]
            description = data["description"]

            if "aggregateRating" in data.keys():
                rating = data["aggregateRating"]["ratingCount"]
            else:
                rating = None

            product_url = url.replace(self.BASE_URL, "")
            price = float(data["offers"]["price"])

            df = pd.DataFrame(
                {
                    "shop": "Jollyes",
                    "name": product_title,
                    "rating": rating,
                    "description": description,
                    "url": product_url,
                    "price": price,
                    "image_urls": ', '.join(data['image']),
                    "variant": None,
                    "discounted_price": None,
                    "discount_percentage": None
                }, index=[0]
            )

            return df

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
