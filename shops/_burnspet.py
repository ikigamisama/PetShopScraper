import asyncio
import requests
import pandas as pd
from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class BurnsPetETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "BurnsPet"
        self.BASE_URL = "https://burnspet.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = 'div.productbig'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        category_link = f"{self.BASE_URL}{category}"

        urls = []
        page = 1

        while True:
            if page == 1:
                category_link_page = category_link
            else:
                category_link_page = f"{category_link}/?paged={page}"

            # Parse request response
            soup = asyncio.run(self.scrape(
                category_link_page, '.productlist-products'))
            if soup:
                links = soup.select(
                    "a[class*='home-productrange-slider-item __productlist']")
                links = [link["href"] for link in links if link.find(
                    'p', class_="home-productrange-slider-item-flavour")]
                urls.extend(links)
                page += 1

            else:
                break

        if urls:
            df = pd.DataFrame({"url": urls})
            df.insert(0, "shop", self.SHOP)

            return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            if soup.find(string="Out of stock"):
                logger.info(f"Skipping {url} as it is sold out. ")
                return None

            product_name = soup.find('div', class_="usercontent").find('h1').get_text(
            ) + ' - ' + soup.find('div', class_="usercontent").find('h2').get_text()
            product_url = url.replace(self.BASE_URL, "")
            product_description = soup.find_all('div', class_="producttabpanel-panel")[
                0].find('div', class_='usercontent').get_text(strip=True)
            rating_wrapper = soup.find_all('div', class_="producttabpanel-panel")[len(soup.find_all(
                'div', class_="producttabpanel-panel")) - 1].find('div', class_="trustpilot-widget")
            product_rating = ''

            if rating_wrapper:
                business_unit_id = rating_wrapper.get(
                    "data-businessunit-id", "")
                template_id = rating_wrapper.get("data-template-id", "")
                locale = rating_wrapper.get("data-locale", "en-GB")
                sku = rating_wrapper.get("data-sku", "")

                sku_encoded = "%2C".join(sku.split(",")) if sku else ""

                get_rating_details = requests.get(
                    f"https://widget.trustpilot.com/trustbox-data/{template_id}?businessUnitId={business_unit_id}&locale={locale}&sku={sku_encoded}")
                if get_rating_details.status_code == 200:
                    if get_rating_details.json()["productReviewsSummary"]["starsAverage"] == 0.0:
                        product_rating = '0/5'
                    else:
                        product_rating = str(get_rating_details.json()[
                                             "productReviewsSummary"]["starsAverage"]) + '/5'
                else:
                    product_rating = '0/5'

            else:
                product_rating = '0/5'

            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            variant_wrapper = soup.find(
                'select', id="Variants").find_all('option')

            if (soup.find('select', id="Variants")):
                for variant in variant_wrapper:
                    variants.append(variant.get_text(strip=True).split('-')[0])
                    prices.append(float(variant.get_text(
                        strip=True).split('-')[1].replace('£', '')))
                    discounted_prices.append(None)
                    discount_percentages.append(None)
                    image_urls.append(
                        soup.find('img', class_="productbig-img").get('src'))

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
