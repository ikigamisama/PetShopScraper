import random
import requests
import asyncio
import nest_asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log
)
from fp.fp import FreeProxy
from loguru import logger

nest_asyncio.apply()

# Configuration constants
MAX_RETRIES = 3
MAX_WAIT_BETWEEN_REQ = 8
MIN_WAIT_BETWEEN_REQ = 2
REQUEST_TIMEOUT = 30000
PAGE_LOAD_TIMEOUT = 30000
PROXY_VALIDATION_TIMEOUT = 5
MAX_PROXY_RETRIES = 10
BROWSER_RESTART_INTERVAL = 20
PROXY_CACHE_SIZE = 50


@dataclass
class ProxyInfo:
    """Data class to store proxy information"""
    proxy: str
    last_used: float
    success_count: int = 0
    failure_count: int = 0
    is_working: bool = True

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class SkipScrape(Exception):
    """Raised to indicate that scraping should be skipped (e.g. 404)."""
    pass


class ScrapingError(Exception):
    """General scraping error that should trigger retries"""
    pass


class ProxyRotator:
    """Enhanced proxy management with rotation and validation"""

    def __init__(self, cache_size: int = PROXY_CACHE_SIZE):
        self.proxies: List[ProxyInfo] = []
        self.cache_size = cache_size
        self.current_index = 0
        self._lock = asyncio.Lock()
        self.last_refresh = 0
        self.refresh_interval = 300  # 5 minutes

    async def get_fresh_proxies(self) -> List[str]:
        """Get fresh proxies from multiple sources"""
        proxy_sources = []

        # Source 1: FreeProxy
        try:
            proxy = FreeProxy(rand=True, timeout=2).get()
            if proxy:
                proxy_sources.append(proxy)
        except Exception as e:
            logger.warning(f"FreeProxy failed: {e}")

        # Source 2: Free proxy list scraping
        try:
            free_proxies = await self._scrape_free_proxy_list()
            proxy_sources.extend(free_proxies[:10])  # Limit to 10
        except Exception as e:
            logger.warning(f"Free proxy list scraping failed: {e}")

        return proxy_sources

    async def _scrape_free_proxy_list(self) -> List[str]:
        """Scrape proxies from free-proxy-list.net"""
        try:
            response = requests.get(
                "https://www.proxy-list.download/api/v1/get?type=http", timeout=10)
            if response.status_code == 200:
                proxies = response.text.strip().split('\n')
                return [f"http://{proxy.strip()}" for proxy in proxies if proxy.strip()]
        except Exception:
            pass

        # Fallback to scraping HTML
        try:
            response = requests.get("https://free-proxy-list.net/", timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            table = soup.find('table', {'id': 'proxylisttable'})
            if not table:
                return []

            proxies = []
            for row in table.find('tbody').find_all('tr')[:20]:  # Limit to 20
                cols = row.find_all('td')
                if len(cols) >= 7 and cols[6].text.strip().lower() == "yes":
                    proxy = f"http://{cols[0].text.strip()}:{cols[1].text.strip()}"
                    proxies.append(proxy)

            return proxies
        except Exception as e:
            logger.error(f"Error scraping free proxy list: {e}")
            return []

    def _validate_proxy_sync(self, proxy: str) -> bool:
        """Synchronous proxy validation"""
        try:
            proxies_dict = {"http": proxy, "https": proxy}
            response = requests.get(
                "http://httpbin.org/ip",
                proxies=proxies_dict,
                timeout=PROXY_VALIDATION_TIMEOUT,
                headers={'User-Agent': UserAgent().random}
            )
            return response.status_code == 200
        except Exception:
            return False

    async def validate_proxies_batch(self, proxy_list: List[str]) -> List[str]:
        """Validate multiple proxies concurrently"""
        valid_proxies = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor, self._validate_proxy_sync, proxy)
                for proxy in proxy_list
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for proxy, is_valid in zip(proxy_list, results):
                if is_valid is True:
                    valid_proxies.append(proxy)

        logger.info(
            f"Validated {len(valid_proxies)} out of {len(proxy_list)} proxies")
        return valid_proxies

    async def refresh_proxy_pool(self):
        """Refresh the proxy pool with new proxies"""
        async with self._lock:
            current_time = time.time()
            if current_time - self.last_refresh < self.refresh_interval:
                return

            logger.info("Refreshing proxy pool...")
            fresh_proxies = await self.get_fresh_proxies()

            if fresh_proxies:
                valid_proxies = await self.validate_proxies_batch(fresh_proxies)

                # Add new valid proxies
                for proxy in valid_proxies:
                    if not any(p.proxy == proxy for p in self.proxies):
                        self.proxies.append(
                            ProxyInfo(proxy=proxy, last_used=0))

                # Remove failed proxies and keep only the best ones
                self.proxies = [p for p in self.proxies if p.success_rate >
                                0.3 or p.success_count + p.failure_count < 3]
                self.proxies = sorted(self.proxies, key=lambda x: x.success_rate, reverse=True)[
                    :self.cache_size]

                self.last_refresh = current_time
                logger.info(
                    f"Proxy pool refreshed. Current pool size: {len(self.proxies)}")

    async def get_proxy(self) -> Optional[str]:
        """Get next available proxy with rotation"""
        if not self.proxies:
            await self.refresh_proxy_pool()

        if not self.proxies:
            logger.warning("No proxies available")
            return None

        async with self._lock:
            # Sort by success rate and last used time
            available_proxies = [p for p in self.proxies if p.is_working]
            if not available_proxies:
                # Reset all proxies if none are working
                for p in self.proxies:
                    p.is_working = True
                available_proxies = self.proxies

            # Select proxy (round-robin with preference for better performing ones)
            proxy_info = available_proxies[self.current_index % len(
                available_proxies)]
            self.current_index = (self.current_index +
                                  1) % len(available_proxies)

            proxy_info.last_used = time.time()
            return proxy_info.proxy

    async def mark_proxy_result(self, proxy: str, success: bool):
        """Mark proxy as successful or failed"""
        async with self._lock:
            for proxy_info in self.proxies:
                if proxy_info.proxy == proxy:
                    if success:
                        proxy_info.success_count += 1
                    else:
                        proxy_info.failure_count += 1
                        if proxy_info.failure_count > 3:
                            proxy_info.is_working = False
                    break


class WebScraper:
    def __init__(self):
        self.ua = UserAgent()
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.playwright_instance = None
        self.pages_scraped = 0
        self.restart_browser_every = BROWSER_RESTART_INTERVAL
        self.proxy_rotator = ProxyRotator()
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

    async def setup_browser(self, browser_type: str = "firefox") -> None:
        """Initialize browser with enhanced configuration"""
        if self.browser is None or self.pages_scraped >= self.restart_browser_every:
            await self.close_browser()

            self.playwright_instance = await async_playwright().start()

            # Get proxy
            self.current_proxy = await self.proxy_rotator.get_proxy()
            proxy_settings = {
                "proxy": {"server": self.current_proxy}} if self.current_proxy else {}

            if self.current_proxy:
                logger.info(f"Using proxy: {self.current_proxy}")

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

            self.pages_scraped = 0

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
        timeout: int = REQUEST_TIMEOUT,
        wait_until: str = "domcontentloaded",
        simulate_behavior: bool = True,
        headers: Optional[Dict[str, str]] = None,
        browser: str = 'firefox'
    ) -> BeautifulSoup:

        page = None
        try:
            await self.setup_browser(browser)

            if not self.context:
                raise ScrapingError("Failed to initialize browser context")

            page = await self.context.new_page()
            page.set_default_timeout(timeout)
            page.set_default_navigation_timeout(PAGE_LOAD_TIMEOUT)

            # Set additional headers if provided
            if headers:
                await page.set_extra_http_headers(self.get_headers(headers))

            logger.info(f"Navigating to: {url}")

            # Navigate with error handling
            try:
                response = await page.goto(url, wait_until=wait_until, timeout=PAGE_LOAD_TIMEOUT)
            except Exception as e:
                if "net::ERR_PROXY_CONNECTION_FAILED" in str(e) and self.current_proxy:
                    await self.proxy_rotator.mark_proxy_result(self.current_proxy, False)
                    raise ScrapingError(f"Proxy connection failed: {e}")
                raise

            if not response:
                raise ScrapingError(f"No response received for {url}")

            status = response.status
            if status == 404:
                raise SkipScrape(f"Page not found (404): {url}")
            elif status >= 400:
                if self.current_proxy:
                    await self.proxy_rotator.mark_proxy_result(self.current_proxy, False)
                raise ScrapingError(f"HTTP {status} error for {url}")

            # Mark proxy as successful if we got a good response
            if self.current_proxy and status < 400:
                await self.proxy_rotator.mark_proxy_result(self.current_proxy, True)

            if simulate_behavior:
                await self.simulate_human_behavior(page, url)

            # Wait for selector with better error handling
            try:
                logger.info(f"Waiting for selector: {selector}")
                await page.wait_for_selector(selector, timeout=timeout)
            except Exception as e:
                # Try alternative selectors or continue without waiting
                logger.warning(
                    f"Selector '{selector}' not found, continuing anyway: {e}")

            # Extract content
            logger.info("Extracting page content...")
            rendered_html = await page.content()
            soup = BeautifulSoup(rendered_html, "html.parser")

            self.pages_scraped += 1
            logger.success(
                f"Successfully extracted content from {url} (Pages scraped: {self.pages_scraped})")

            return soup

        except asyncio.TimeoutError as e:
            if self.current_proxy:
                await self.proxy_rotator.mark_proxy_result(self.current_proxy, False)
            logger.error(f"Timeout for {url}: {e}")
            raise ScrapingError(f"Timeout for {url}: {e}")

        except Exception as e:
            if self.current_proxy and "proxy" in str(e).lower():
                await self.proxy_rotator.mark_proxy_result(self.current_proxy, False)
            logger.error(f"Error scraping {url}: {str(e)}")
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
        timeout: int = REQUEST_TIMEOUT,
        wait_until: str = "domcontentloaded",
        simulate_behavior: bool = True,
        headers: Optional[Dict[str, str]] = None,
        browser: str = "firefox"
    ) -> Optional[BeautifulSoup]:
        try:
            return await retry_extract_scrape_content(
                self, url, selector, timeout, wait_until, simulate_behavior, headers, browser
            )
        except SkipScrape as e:
            logger.warning(f"Skipping scrape: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to scrape after {MAX_RETRIES} attempts: {e}")
            return None

    async def close_browser(self):
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

    async def close(self, force: bool = False):
        """Clean up all resources"""
        await self.close_browser()


@retry(
    wait=wait_exponential(
        multiplier=1, min=MIN_WAIT_BETWEEN_REQ, max=MAX_WAIT_BETWEEN_REQ),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(ScrapingError),
    before_sleep=before_sleep_log(logger, "WARNING"),
    reraise=True,
)
async def retry_extract_scrape_content(scraper, *args, **kwargs):
    return await scraper._extract_scrape_content(*args, **kwargs)


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
    headers: Optional[Dict[str, str]] = None,
    wait_until: str = "domcontentloaded",
    min_sec: float = 2,
    max_sec: float = 5,
    browser: str = 'firefox'
) -> Optional[BeautifulSoup]:
    """Scrape a single URL with enhanced error handling"""
    async with AsyncWebScraper() as scraper:
        result = await scraper.extract_scrape_content(
            url, selector, headers=headers, wait_until=wait_until, browser=browser
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


async def scrape_urls_batch(
    urls_and_selectors: List[Tuple[str, str]],
    max_concurrent: int = 3,
    delay_range: Tuple[float, float] = (2, 5)
) -> List[Optional[BeautifulSoup]]:
    """Scrape multiple URLs with concurrency control"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def scrape_with_semaphore(url: str, selector: str):
        async with semaphore:
            async with AsyncWebScraper() as scraper:
                result = await scraper.extract_scrape_content(url, selector)
                await asyncio.sleep(random.uniform(*delay_range))
                return result

    tasks = [scrape_with_semaphore(url, selector)
             for url, selector in urls_and_selectors]
    return await asyncio.gather(*tasks, return_exceptions=True)


# Legacy function for backward compatibility
async def scrape_urls(urls_and_selectors: List[Tuple[str, str]]) -> List[Optional[BeautifulSoup]]:
    """Legacy function - use scrape_urls_batch instead"""
    return await scrape_urls_batch(urls_and_selectors, max_concurrent=1)


def random_proxy_free():
    """Legacy function - kept for backward compatibility"""
    try:
        proxy_rotator = ProxyRotator()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(proxy_rotator.get_proxy())
    except Exception as e:
        logger.error(f"Error getting random proxy: {e}")
        return None
