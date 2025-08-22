import math
import asyncio
import pandas as pd


from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class PetDrugsOnlineETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "PetDrugsOnline"
        self.BASE_URL = "https://www.petdrugsonline.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '.page-columns'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        current_url = f"{self.BASE_URL}{category}"
        urls = []

        soup = asyncio.run(self.scrape(
            current_url, self.SELECTOR_SCRAPE_PRODUCT_INFO))

        if not soup:
            logger.error(
                f"[ERROR] Initial scrape returned no content for URL: {current_url}")
            return pd.DataFrame(columns=["shop", "url"])

        try:
            amount_spans = soup.find('p', id='toolbar-amount').find_all('span')
            initial_number_product = int(amount_spans[1].get_text())
            all_product_number = int(amount_spans[2].get_text())
        except Exception as e:
            logger.error(
                f"[ERROR] Failed to extract product counts from {current_url}: {e}")
            return pd.DataFrame(columns=["shop", "url"])

        times_to_click = math.ceil(all_product_number / initial_number_product)

        for i in range(1, times_to_click + 1):
            page_url = f"{current_url}?p={i}"
            page_pagination_soup = asyncio.run(self.scrape(
                page_url, self.SELECTOR_SCRAPE_PRODUCT_INFO))

            if not page_pagination_soup:
                logger.warning(
                    f"[WARNING] Skipping page {i} due to empty or invalid soup.")
                continue

            try:
                product_list_container = page_pagination_soup.find(
                    "ol", class_="products list items product-items")
                if not product_list_container:
                    logger.warning(
                        f"[WARNING] Product list container not found on page {i}.")
                    continue

                product_list = product_list_container.find_all('li')
                urls.extend([
                    a_tag.get('href') for product in product_list
                    if (a_tag := product.find('a')) and a_tag.get('href')
                ])
            except Exception as e:
                logger.error(
                    f"[ERROR] Failed to extract product list from {page_url}: {e}")
                continue

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)
        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_name = soup.find(
                'h1', class_="page-title").find('span').get_text()
            product_url = url.replace(self.BASE_URL, "")

            product_description = " ".join([p.get_text(strip=True) for p in soup.find('div', class_="product-attribute-description")
                                            .find('div', class_="product-attribute-value")
                                            .find_all(['p', 'strong'])])
            product_rating = soup.find(
                'span', class_='review-summary-rating-text').get_text(strip=True)

            variant_wrapper = soup.find(
                'ul', id='custom-select-attribute-results').find_all('li')
            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            for variant in variant_wrapper:
                variants.append(variant.find(
                    'span', class_="custom-option-col-label").get_text(strip=True))
                prices.append(float(variant.find(
                    'span', class_="price-wrapper").find('span').get_text().replace('£', '')))
                image_urls.append(
                    soup.find('div', class_="product-gallery").find('img').get('src'))

                if (variant.find('span', class_="custom-option-col-inner").get_text(strip=True) != ""):
                    previous_price = float(variant.find('span', class_="custom-option-col-inner").find(
                        'span', class_='vet-price').find('span', class_='price').get_text().replace('£', ''))
                    saving_price = float(variant.find('span', class_="custom-option-col-inner").find(
                        'span', class_='saving-price').find('span', class_='price').get_text().replace('£', ''))

                    discount_percentage = round(
                        (saving_price / previous_price) * 100, 2)
                    discounted_prices.append(saving_price)
                    discount_percentages.append(discount_percentage)
                else:
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
