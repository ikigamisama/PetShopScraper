import re
import math
import json
import asyncio
import pandas as pd
from functions.etl import PetProductsETL
from bs4 import BeautifulSoup
from loguru import logger


class DirectVetETL(PetProductsETL):
    def __init__(self):
        super().__init__()
        self.SHOP = "DirectVet"
        self.BASE_URL = "https://www.direct-vet.co.uk"
        self.SELECTOR_SCRAPE_PRODUCT_INFO = '#center_column'
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 2

    def extract(self, category):
        current_url = f"{self.BASE_URL}/{category}"
        urls = []

        soup = asyncio.run(self.scrape(
            current_url, self.SELECTOR_SCRAPE_PRODUCT_INFO, wait_until="networkidle"))

        # Check soup is valid and not a boolean
        if not soup or isinstance(soup, bool):
            logger.error(
                f"[ERROR] Failed to scrape category page: {current_url}")
            return pd.DataFrame(columns=["shop", "url"])

        # Check if category has no products
        heading_counter = soup.find('small', class_="heading-counter")
        if heading_counter and 'There are no products in this category' in heading_counter.get_text():
            logger.info(f"[INFO] No products found in category: {category}")
            return pd.DataFrame(columns=["shop", "url"])

        # Parse number of products
        try:
            product_text = heading_counter.get_text()
            product_count = int(
                re.sub(r"There (is|are) | products\.| product\.", "", product_text))
        except Exception as e:
            logger.warning(f"[WARN] Could not parse product count: {e}")
            product_count = 0

        pagination_page_num = math.ceil(product_count / 12)

        for i in range(1, pagination_page_num + 1):
            page_url = f"{current_url}?selected_filters=page-{i}"

            page_pagination_source = asyncio.run(
                self.scrape(page_url, self.SELECTOR_SCRAPE_PRODUCT_INFO, wait_until="networkidle"))

            if not page_pagination_source or isinstance(page_pagination_source, bool):
                logger.warning(
                    f"[WARN] Skipped page {i} due to failed scrape: {page_url}")
                continue

            for link in page_pagination_source.find_all('a', class_="product_img_link"):
                if link.get('href'):
                    urls.append(link.get('href'))

        df = pd.DataFrame({"url": urls})
        df.insert(0, "shop", self.SHOP)
        return df

    def transform(self, soup: BeautifulSoup, url: str):
        try:
            product_name = soup.find('h1', itemprop="name").get_text()
            product_url = url.replace(self.BASE_URL, "")
            product_description = soup.find(
                'div', id="short_description_content").get_text(strip=True)
            product_rating = ''

            rating_wrapper = soup.find(
                'div', id="product_comments_block_extra").find('div', 'star_content')
            if (rating_wrapper):
                rating_list_wrapper = soup.find('div', id="product_comments_block_tab").find_all(
                    'div', itemprop="reviewRating")
                rate_list = [int(rating.find('meta', itemprop="ratingValue").get(
                    'content')) for rating in rating_list_wrapper]

                avg_rating = round(sum(rate_list) / len(rate_list), 2)
                product_rating = f"{int(avg_rating) if avg_rating.is_integer() else avg_rating}/5"
            else:
                product_rating = '0/5'

            variant_wrapper = soup.find('table', id='ct_matrix')
            variants = []
            prices = []
            discounted_prices = []
            discount_percentages = []
            image_urls = []

            if (variant_wrapper):
                for variant in variant_wrapper.find('tbody').find_all('tr'):
                    variant_info = ''
                    if (variant.find('td', attrs={'data-label': "Select"})):
                        variant_info = variant.find(
                            'td', attrs={'data-label': "Select"}).get_text()
                    elif (variant.find('td', attrs={'data-label': "Color"})):
                        variant_info = variant.find(
                            'td', attrs={'data-label': "Color"}).get_text()
                    else:
                        variant_info = variant.find(
                            'td', attrs={'data-label': "Size"}).get_text()

                    variants.append(variant_info)
                    image_urls.append(soup.find('img', id="bigpic").get('src'))

                    if (variant.find('td', attrs={'data-label': "Price"}).find('strike')):
                        former_price = float(variant.find(
                            'td', attrs={'data-label': "Price"}).find('strike').get_text().replace('£', ''))
                        current_price = float(variant.find('td', attrs={'data-label': "Price"}).find(
                            'strong', class_="strongprice").get_text().replace('£', ''))

                        discounted_prices.append(former_price - current_price)
                        discount_percentages.append(
                            round(((former_price - current_price) / former_price) * 100, 2))
                        prices.append(float(current_price))

                    else:
                        prices.append(float(variant.find(
                            'td', attrs={'data-label': "Price"}).get_text().replace('£', '')))
                        discounted_prices.append(None)
                        discount_percentages.append(None)

            else:
                variant_info = ''
                description_div = soup.find(
                    'div', id="short_description_content")
                if description_div:
                    h2_tag = description_div.find('h2')
                    if h2_tag:
                        variant_info = h2_tag.get_text(strip=True).replace(
                            '- ', '').replace('-', '').strip()
                    else:
                        p_tags = description_div.find_all('p')
                        if p_tags:
                            variant_info = p_tags[-1].get_text(strip=True).replace(
                                '- ', '').replace('-', '').strip()
                        else:
                            variant_info = None
                else:
                    variant_info = None

                variants.append(variant_info)
                prices.append(
                    float(soup.find('span', itemprop="price").get_text().replace('£', '')))
                discounted_prices.append(None)
                discount_percentages.append(None)
                image_urls.append(soup.find('img', id="bigpic").get('src'))

            df = pd.DataFrame({"variant": variants, "price": prices, "discounted_price": discounted_prices,
                              "discount_percentage": discount_percentages, "image_urls": image_urls})
            df.insert(0, "url", product_url)
            df.insert(0, "description", product_description)
            df.insert(0, "rating", product_rating)
            df.insert(0, "name", product_name)
            df.insert(0, "shop", self.SHOP)

            return df

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
