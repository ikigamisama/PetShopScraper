
import asyncio
import requests
import time

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List, Tuple
from fp.fp import FreeProxy
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from loguru import logger

PROXY_CACHE_SIZE = 50
PROXY_VALIDATION_TIMEOUT = 5


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
