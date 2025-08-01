import random

import asyncio
import nest_asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from fake_useragent import UserAgent
from .proxy import ProxyRotator
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log
)

from loguru import logger

nest_asyncio.apply()

# Configuration constants
MAX_RETRIES = 5
MAX_WAIT_BETWEEN_REQ = 8
MIN_WAIT_BETWEEN_REQ = 2
REQUEST_TIMEOUT = 60000
PAGE_LOAD_TIMEOUT = 60000

MAX_PROXY_RETRIES = 10
BROWSER_RESTART_INTERVAL = 20


class SkipScrape(Exception):
    """Raised to indicate that scraping should be skipped (e.g. 404)."""
    pass


class ScrapingError(Exception):
    """General scraping error that should trigger retries"""
    pass


class WebScraper:
    def __init__(self):
        self.ua = UserAgent()
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.playwright_instance = None
        self.pages_scraped = 0
        self.restart_browser_every = BROWSER_RESTART_INTERVAL
        self.current_proxy = None

    def get_headers(self, headers=None) -> Dict[str, str]:
        """Generate realistic browser headers with better randomization"""
        user_agent = self.ua.random

        # Browser-specific headers based on user agent
        if 'Chrome' in user_agent:
            sec_ch_ua = '"Not.A/Brand";v="24", "Chromium";v="122", "Google Chrome";v="122"'
        elif 'Firefox' in user_agent:
            sec_ch_ua = '"Not.A/Brand";v="24", "Firefox";v="122"'
        else:
            sec_ch_ua = '"Not.A/Brand";v="24", "Chromium";v="122"'

        default_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": random.choice(["max-age=0", "no-cache"]),
            "User-Agent": user_agent,
            "Priority": "u=0, i",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Ch-Ua": sec_ch_ua,
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": random.choice(["none", "same-origin", "cross-site"]),
            "Sec-Fetch-User": "?1"
        }

        if headers:
            default_headers.update(headers)

        return default_headers

    async def setup_browser(self, proxy, browser_type: str = "firefox") -> None:
        """Initialize browser with enhanced configuration"""
        self.browser = None

        self.playwright_instance = await async_playwright().start()

        logger.info(f"Using proxy {proxy}")

        # Get proxy
        proxy_settings = {"proxy": {"server": proxy}} if proxy else {}

        # Enhanced browser arguments
        stealth_args = [
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-sync",
            "--disable-extensions",
            "--disable-popup-blocking",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor",
            "--disable-ipc-flooding-protection",
        ]

        if browser_type == "firefox":
            self.browser = await self.playwright_instance.firefox.launch(
                headless=True,
                args=stealth_args + ["--no-remote"],
                firefox_user_prefs={
                    # Performance optimizations
                    "permissions.default.image": 2,
                    "browser.cache.disk.enable": False,
                    "browser.cache.memory.enable": False,
                    "media.autoplay.enabled": False,
                    "media.video_stats.enabled": False,

                    # Anti-detection
                    "dom.webdriver.enabled": False,
                    "media.navigator.enabled": False,
                    "webgl.disabled": True,
                    "privacy.trackingprotection.enabled": True,
                    "geo.enabled": False,
                    "general.platform.override": "Win32",
                    "general.appversion.override": "5.0 (Windows)",
                    "general.oscpu.override": "Windows NT 10.0; Win64; x64",

                    # Network optimizations
                    "network.http.pipelining": True,
                    "network.http.pipelining.maxrequests": 8,
                    "network.http.max-connections": 32,
                },
                **proxy_settings
            )
        else:
            self.browser = await self.playwright_instance.chromium.launch(
                headless=True,
                args=stealth_args + [
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                    "--disable-notifications",
                ],
                **proxy_settings
            )

        if self.context is None:
            context_options = {
                "locale": random.choice(["en-US", "en-GB", "en-CA"]),
                "user_agent": self.ua.random,
                "viewport": {"width": random.randint(1366, 1920), "height": random.randint(768, 1080)},
                "java_script_enabled": True,
                "ignore_https_errors": True,
                "extra_http_headers": self.get_headers(),
                "timezone_id": random.choice(["America/New_York", "Europe/London", "America/Los_Angeles"]),
                "permissions": [],  # Minimize permissions
            }

            self.context = await self.browser.new_context(**context_options)

            # Enhanced request interception
            await self.context.route("**/*", self._route_handler)

    async def _route_handler(self, route):
        """Enhanced route handler for blocking unwanted resources"""
        url = route.request.url
        resource_type = route.request.resource_type

        # Block unwanted resources
        block_patterns = [
            'analytics', 'ads', 'tracking', 'metrics', 'telemetry',
            'facebook.com', 'google-analytics', 'googletagmanager',
            'doubleclick.net', 'adsystem.com', 'amazon-adsystem.com'
        ]

        block_types = ['image', 'media', 'font', 'other']

        if any(pattern in url.lower() for pattern in block_patterns) or resource_type in block_types:
            await route.abort()
        else:
            await route.continue_()

    async def simulate_human_behavior(self, page: Page, url: str):
        """Enhanced human behavior simulation"""
        # Random delay
        await asyncio.sleep(random.uniform(0.8, 2.0))

        # Random mouse movements
        for _ in range(random.randint(1, 3)):
            await page.mouse.move(
                random.randint(100, 1200),
                random.randint(100, 800)
            )
            await asyncio.sleep(random.uniform(0.1, 0.3))

        # Random scrolling
        if random.random() < 0.4:
            scroll_amount = random.randint(100, 800)
            await page.mouse.wheel(0, scroll_amount)
            await asyncio.sleep(random.uniform(0.5, 1.0))

        # Random click (sometimes)
        if random.random() < 0.1:
            try:
                await page.mouse.click(random.randint(200, 800), random.randint(200, 600))
                await asyncio.sleep(random.uniform(0.2, 0.8))
            except Exception:
                pass  # Ignore click errors

    async def _extract_scrape_content(
        self,
        url: str,
        selector: str,
        proxy: str,
        timeout: int = REQUEST_TIMEOUT,
        wait_until: str = "domcontentloaded",
        simulate_behavior: bool = True,
        headers: Optional[Dict[str, str]] = None,
        browser: str = 'firefox',

    ) -> BeautifulSoup:

        page = None
        try:
            await self.setup_browser(proxy, browser)

            if not self.context:
                raise ScrapingError("Failed to initialize browser context")

            page = await self.context.new_page()
            page.set_default_timeout(timeout)
            page.set_default_navigation_timeout(PAGE_LOAD_TIMEOUT)

            # Set additional headers if provided
            if headers:
                await page.set_extra_http_headers(self.get_headers(headers))

            logger.info(f"Navigating to: {url}")
            await page.goto(url, wait_until=wait_until, timeout=PAGE_LOAD_TIMEOUT)

            logger.info(f"Waiting for selector: {selector}")
            await page.wait_for_selector(selector, timeout=timeout)

            # Extract content
            logger.info("Extracting page content...")
            rendered_html = await page.content()
            soup = BeautifulSoup(rendered_html, "html.parser")

            self.pages_scraped += 1
            logger.success(
                f"Successfully extracted content from {url}")

            return soup

        except asyncio.TimeoutError as e:
            raise ScrapingError(f"Timeout for {url}: {e}")

        except Exception as e:
            raise ScrapingError(f"Error scraping {url}: {str(e)}")

        finally:
            if page:
                try:
                    await page.close()
                except Exception as e:
                    logger.error(f"Error closing page: {e}")

    async def extract_scrape_content(
        self,
        url: str,
        selector: str,
        proxy: bool,
        timeout: int = REQUEST_TIMEOUT,
        wait_until: str = "domcontentloaded",
        simulate_behavior: bool = True,
        headers: Optional[Dict[str, str]] = None,
        browser: str = "firefox"

    ) -> Optional[BeautifulSoup]:

        try:
            return await retry_extract_scrape_content(
                self, url, selector, proxy, timeout, wait_until, simulate_behavior, headers, browser
            )
        except SkipScrape as e:
            logger.warning(f"Skipping scrape: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to scrape after {MAX_RETRIES} attempts: {e}")
            return None

    async def close(self):
        """Close only browser resources, keep proxy rotator"""
        try:
            if self.context:
                await self.context.close()
                self.context = None

            if self.browser:
                await self.browser.close()
                self.browser = None

            if self.playwright_instance:
                await self.playwright_instance.stop()
                self.playwright_instance = None

            logger.info("Browser resources closed")

        except Exception as e:
            logger.error(f"Error during browser close: {e}")


@retry(
    wait=wait_exponential(
        multiplier=1, min=MIN_WAIT_BETWEEN_REQ, max=MAX_WAIT_BETWEEN_REQ),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(ScrapingError),
    before_sleep=before_sleep_log(logger, "WARNING"),
    reraise=True,
)
async def retry_extract_scrape_content(scraper, url, selector, proxy, timeout, wait_until, simulate_behavior, headers, browser):
    generate_proxy = await ProxyRotator().get_proxy() if proxy == True else ''
    return await scraper._extract_scrape_content(url, selector, generate_proxy, timeout, wait_until, simulate_behavior, headers, browser)


class AsyncWebScraper:
    def __init__(self):
        self.scraper = WebScraper()

    async def __aenter__(self):
        return self.scraper

    async def __aexit__(self, *_):
        await self.scraper.close()


# Enhanced convenience functions
async def scrape_url(
    url: str,
    selector: str,
    proxy: bool,
    headers: Optional[Dict[str, str]] = None,
    wait_until: str = "domcontentloaded",
    min_sec: float = 2,
    max_sec: float = 5,
    browser: str = 'firefox'
) -> Optional[BeautifulSoup]:
    """Scrape a single URL with enhanced error handling"""
    async with AsyncWebScraper() as scraper:
        result = await scraper.extract_scrape_content(
            url, selector, proxy, headers=headers, wait_until=wait_until, browser=browser
        )

        # Smart delay based on success
        if result is not None:
            delay = random.uniform(min_sec, max_sec)
        else:
            # Longer delay on failure
            delay = random.uniform(max_sec, max_sec * 2)

        if delay >= 60:
            minutes = int(delay // 60)
            seconds = delay % 60
            logger.info(f"Sleep for {minutes} min {seconds:.2f} sec")
        else:
            logger.info(f"Sleep for {delay:.2f} sec")

        await asyncio.sleep(delay)
        return result
