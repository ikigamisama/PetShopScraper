import os
import json
import asyncio
import pandas as pd

from abc import ABC, abstractmethod
from sqlalchemy.engine import Engine
from .connection import Connection
from .scraper import scrape_url
from .proxy import ProxyRotator
from loguru import logger
from datetime import datetime as dt
from bs4 import BeautifulSoup


class PetProductsETL(ABC):
    def __init__(self):
        self.SHOP = ""
        self.BASE_URL = ""
        self.SELECTOR_SCRAPE_PRODUCT_INFO = ''
        self.MIN_SEC_SLEEP_PRODUCT_INFO = 1
        self.MAX_SEC_SLEEP_PRODUCT_INFO = 3
        self.connection = Connection()
        self.wait_until = "load"
        self.browser_type = 'chromium'
        self.with_proxy = False

    async def scrape(self, url, selector, proxy, headers=None, wait_until="load", min_sec=2, max_sec=5, browser="firefox"):
        soup = await scrape_url(url, selector, proxy, headers, wait_until, min_sec=min_sec, max_sec=max_sec, browser=browser)
        return soup if soup else False

    @abstractmethod
    def extract(self, category):
        pass

    @abstractmethod
    def transform(self, soup: BeautifulSoup, url: str):
        pass

    def load(self, data: pd.DataFrame, table_name: str):
        try:
            n = data.shape[0]
            data.to_sql(table_name, self.connection.engine,
                        if_exists="append", index=False)
            logger.success(
                f"Successfully loaded {n} records to the {table_name}.")

        except Exception as e:
            logger.error(e)
            raise e

    def extract_unscraped_data(self, temp_table):
        create_temp_sql = self.connection.get_sql_from_file(
            'create_temp_table_product_info.sql')
        create_temp_sql = create_temp_sql.format(
            table_name=temp_table)

        self._temp_table(create_temp_sql, temp_table, 'created')

        sql = self.connection.get_sql_from_file('select_unscraped_urls.sql')
        sql = sql.format(shop=self.SHOP, table_name="urls")

        return self.connection.extract_from_sql(sql)

    def insert_scrape_in_database(self, temp_table):
        for sql_file, label in [
            ('insert_into_pet_products.sql', 'data product inserted'),
            ('insert_into_pet_product_variants.sql',
             'data product variant inserted'),
            ('insert_into_pet_product_variant_prices.sql',
             'data product price inserted')
        ]:
            sql = self.connection.get_sql_from_file(
                sql_file).format(table_name=temp_table)
            self._temp_table(sql, temp_table, label)

        self._temp_table(f"DROP TABLE {temp_table};", temp_table, 'deleted')

    def get_product_infos(self):
        temp_table = f"stg_{self.SHOP.lower()}_temp_products"
        df_urls = self.extract_unscraped_data(temp_table)

        for i, row in df_urls.iterrows():
            pkey = row["id"]
            url = row["url"]

            now = dt.now().strftime("%Y-%m-%d %H:%M:%S")
            soup = asyncio.run(self.scrape(
                url, self.SELECTOR_SCRAPE_PRODUCT_INFO, proxy=self.with_proxy, min_sec=self.MIN_SEC_SLEEP_PRODUCT_INFO, max_sec=self.MAX_SEC_SLEEP_PRODUCT_INFO, wait_until=self.wait_until, browser=self.browser_type))

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

    def get_links_by_category(self):
        self.connection.execute_query(
            f"DELETE FROM urls WHERE shop = '{self.SHOP}'")

        temp_table = f"stg_{self.SHOP.lower()}_temp"
        temp_url_table = f"stg_{self.SHOP.lower()}_temp_url_links"

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(
            BASE_DIR, 'data', 'categories', f'{self.SHOP.lower()}.json')

        if not self.connection.check_table_exists(temp_url_table):

            for sql_file, table in [
                ('create_temp_table_url_links.sql', temp_url_table),
                ('create_temp_table_get_links.sql', temp_table)
            ]:
                sql = self.connection.get_sql_from_file(
                    sql_file).format(table_name=table)
                self._temp_table(sql, table, 'created')

            with open(file_path, 'r+') as f:
                d = json.load(f)
                categories = d['data']

                for value in categories:
                    query = f"""
                        INSERT INTO {temp_url_table} (shop, url, scrape_status, updated_date)
                        VALUES ('{self.SHOP}', '{value}', 'NOT STARTED', '{dt.now()}')
                    """
                    self.connection.execute_query(query)

        sql = self.connection.get_sql_from_file('select_unscraped_urls.sql')
        sql = sql.format(shop=self.SHOP, table_name=temp_url_table)
        df_urls = self.connection.extract_from_sql(sql)

        for i, row in df_urls.iterrows():
            pkey = row["id"]
            url = row["url"]

            now = dt.now().strftime("%Y-%m-%d %H:%M:%S")
            df = self.extract(url)
            if df is not None:
                self.load(df, temp_table)
                self.connection.update_url_scrape_status(
                    pkey, "DONE", temp_url_table, now)
            else:
                self.connection.update_url_scrape_status(
                    pkey, "FAILED", temp_url_table, now)

            logger.info(f"{i+1} out of {len(df_urls)} URL(s) Scraped")

        insert_url_from_temp_sql = self.connection.get_sql_from_file(
            'insert_into_urls.sql').format(table_name=temp_table)
        self._temp_table(insert_url_from_temp_sql, temp_table, 'data inserted')

        for table in [temp_table, temp_url_table]:
            drop_sql = f"DROP TABLE {table};"
            self._temp_table(drop_sql, table, 'deleted')

    def _temp_table(self, sql, table, method):
        self.connection.execute_query(sql)
        logger.info(f"Temporary table {table} {method}.")
