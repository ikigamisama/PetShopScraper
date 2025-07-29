import sys
import argparse
import datetime as dt

from loguru import logger
from functions.factory import SHOPS, run_etl

shop_choice = [i for i in SHOPS.keys()]
PROGRAM_NAME = "Pet Products Scraper"

parser = argparse.ArgumentParser(
    prog=PROGRAM_NAME,
    description="Scrape product details from various pet shops."
)

parser.add_argument("task", choices=[
                    "get_links", "scrape"], help="Identify the task to be executed. get_links=get links from registered shops; scrape=scrape products.")
parser.add_argument("-s", "--shop", choices=shop_choice,
                    help="Select a shop to scrape. Default: all shops.")
args = parser.parse_args()

if __name__ == "__main__":
    start_time = dt.datetime.now()
    logger.remove()
    logger.add("logs/std_out.log", rotation="10 MB", level="INFO")
    logger.add("logs/std_err.log", rotation="10 MB", level="ERROR")
    logger.add(sys.stdout, level="INFO")
    logger.add(sys.stderr, level="ERROR")

    logger.info(f"{PROGRAM_NAME} has started")

    task = args.task
    shop = args.shop

    client = run_etl(shop)

    if task == "get_links":
        client.get_links_by_category()

    elif task == "scrape":
        client.get_product_infos()

    end_time = dt.datetime.now()
    duration = end_time - start_time
    logger.info(f"{PROGRAM_NAME} (shop={shop}) has ended. Elapsed: {duration}")
