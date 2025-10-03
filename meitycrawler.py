# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
# from webdriver_manager.chrome import ChromeDriverManager


# from bs4 import BeautifulSoup
# import json
# import time
# import os
# import re
# from urllib.parse import urljoin, urlparse
# from collections import deque, Counter
# import logging
# import requests
# import PyPDF2
# from io import BytesIO


# # NEW: optional language detection
# try:
#     from langdetect import detect as _lang_detect
# except Exception:
#     _lang_detect = None

# # NEW: optional robots.txt
# import urllib.robotparser as robotparser

# # NEW: optional XML for sitemap
# import xml.etree.ElementTree as ET



# class MeitYSeleniumCrawler:
#     def __init__(self, base_url="https://www.meity.gov.in", delay=2, max_pages=500, headless=True,
#                  respect_robots=True, use_sitemap=True):

#         # Create logger
#         self.logger = logging.getLogger(__name__)
#         self.logger.setLevel(logging.INFO)
#         ch = logging.StreamHandler()
#         ch.setLevel(logging.INFO)
#         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#         ch.setFormatter(formatter)
#         self.logger.addHandler(ch)

#         # Attributes
#         self.base_url = base_url
#         self.domain = urlparse(base_url).netloc
#         self.delay = delay
#         self.max_pages = max_pages
#         self.respect_robots = respect_robots
#         self.use_sitemap = use_sitemap

#         self.unwanted_phrases = [
#             "Customize cookies",
#             "This website uses cookies",
#             "Accept all cookies",
#             "Decline optional cookies",
#             "Cookie Settings",
#         ]
#         self.skip_path_keywords = {"/cookies", "/cookie", "/privacy", "/disclaimer", "/terms", "/javascript:", "mailto:"}
#         self.cookies_handled = False

#         # Chrome options
#         self.chrome_options = Options()
#         if headless:
#             self.chrome_options.add_argument('--headless=new')
#         self.chrome_options.add_argument('--no-sandbox')
#         self.chrome_options.add_argument('--disable-dev-shm-usage')
#         self.chrome_options.add_argument('--disable-gpu')
#         self.chrome_options.add_argument('--window-size=1920,1080')
#         self.chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
#         self.chrome_options.add_argument('--log-level=3')
#         self.chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])

#         # Initialize driver
#         self.driver = None
#         self.setup_driver()

#         # Requests session
#         self.session = requests.Session()
#         self.session.headers.update({
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
#         })

#         # Data structures
#         self.visited_urls = set()
#         self.enqueued_urls = set()
#         self.extracted_data = []
#         self.pdf_links = []
#         self.url_queue = deque([base_url])
#         self.enqueued_urls.add(base_url)

#         # Robots.txt
#         self.rp = None
#         if self.respect_robots:
#             self._load_robots()

#         # Sitemap
#         if self.use_sitemap:
#             self._seed_from_sitemap()

#     def _load_robots(self):
#         """Load robots.txt once"""
#         try:
#             robots_url = urljoin(self.base_url, "/robots.txt")
#             self.rp = robotparser.RobotFileParser()
#             self.rp.set_url(robots_url)
#             self.rp.read()
#             self.logger.info(f"Loaded robots.txt from {robots_url}")
#         except Exception as e:
#             self.logger.warning(f"Could not load robots.txt: {e}")
#             self.rp = None

#     def _allowed_by_robots(self, url):
#         if not self.respect_robots or not self.rp:
#             return True
#         try:
#             return self.rp.can_fetch("*", url)
#         except Exception:
#             return True

#     def _seed_from_sitemap(self):
#         """Try to add sitemap URLs to queue for more coverage."""
#         candidates = [
#             "/sitemap.xml",
#             "/sitemap_index.xml",
#             "/sitemap/sitemap.xml",
#         ]
#         seen = 0
#         for path in candidates:
#             try:
#                 sm_url = urljoin(self.base_url, path)
#                 r = self.session.get(sm_url, timeout=15)
#                 if r.status_code != 200 or not r.text.strip():
#                     continue
#                 urls = self._parse_sitemap_xml(r.text, sm_url)
#                 added = 0
#                 for u in urls:
#                     if self.is_valid_url(u) and u not in self.enqueued_urls and self._allowed_by_robots(u):
#                         self.url_queue.append(u)
#                         self.enqueued_urls.add(u)
#                         added += 1
#                 if added:
#                     self.logger.info(f"Seeded {added} URLs from {sm_url}")
#                     seen += added
#             except Exception as e:
#                 self.logger.debug(f"Sitemap seed failed for {path}: {e}")
#         if seen == 0:
#             self.logger.info("No sitemap URLs added (not found or empty).")

#     def _parse_sitemap_xml(self, xml_text, base_url_for_nested):
#         """Parse simple sitemap or sitemap index."""
#         out = []
#         try:
#             root = ET.fromstring(xml_text)
#             ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
#             # sitemap index?
#             for site in root.findall("sm:sitemap", ns):
#                 loc = site.find("sm:loc", ns)
#                 if loc is not None and loc.text:
#                     try:
#                         r = self.session.get(loc.text.strip(), timeout=15)
#                         if r.status_code == 200:
#                             out.extend(self._parse_sitemap_xml(r.text, loc.text.strip()))
#                     except Exception:
#                         continue
#             # url set
#             for url in root.findall("sm:url", ns):
#                 loc = url.find("sm:loc", ns)
#                 if loc is not None and loc.text:
#                     out.append(loc.text.strip())
#         except Exception:
#             # fallback: crude regex if XML namespace tricky
#             out.extend(re.findall(r"<loc>(.*?)</loc>", xml_text, flags=re.I))
#         return list(dict.fromkeys(out))  # unique preserve order

#     def setup_driver(self):
#         """Setup Chrome WebDriver"""
#         try:
#             service = Service(ChromeDriverManager().install())
#             self.driver = webdriver.Chrome(service=service, options=self.chrome_options)
#             self.driver.set_page_load_timeout(40)
#             self.driver.implicitly_wait(10)
#             self.logger.info("WebDriver setup successful")
#         except Exception as e:
#             self.logger.error(f"Error setting up WebDriver: {str(e)}")
#             raise

#     def is_valid_url(self, url):
#         """Check if URL is valid and belongs to MeitY domain"""
#         try:
#             parsed = urlparse(url)
#             if parsed.scheme not in ("http", "https"):
#                 return False
#             # skip certain boilerplate paths
#             lower = parsed.path.lower()
#             for bad in self.skip_path_keywords:
#                 if bad in lower:
#                     return False
#             # within domain?
#             return (parsed.netloc == self.domain or
#                     parsed.netloc == f"www.{self.domain}" or
#                     parsed.netloc == self.domain.replace("www.", ""))
#         except Exception:
#             return False

#     def clean_text(self, text):
#         """Clean extracted text"""
#         if not text:
#             return ""
#         # Remove extra whitespaces and normalize
#         text = re.sub(r'\s+', ' ', text)

#         # Strip common cookie/boilerplate phrases (NEW)
#         for phrase in self.unwanted_phrases:
#             text = text.replace(phrase, ' ')

#         # Remove unwanted characters but keep essential punctuation
#         text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\"\'\/\%\@\#]', '', text)
#         return text.strip()

#     def _handle_cookie_popup_once(self):
#         """Try to accept/close cookie banner once per session (NEW)"""
#         if self.cookies_handled:
#             return
#         try:
#             # common buttons: Accept / Okay / I agree
#             possible_xpaths = [
#                 "//button[contains(translate(., 'ACEPTOKY', 'aceptoky'),'accept')]",
#                 "//button[contains(translate(., 'ACEPTOKY', 'aceptoky'),'ok')]",
#                 "//button[contains(translate(., 'AGREE', 'agree'),'agree')]",
#                 "//button[contains(., 'Accept all cookies')]",
#                 "//a[contains(., 'Accept')]",
#             ]
#             for xp in possible_xpaths:
#                 try:
#                     btn = WebDriverWait(self.driver, 3).until(
#                         EC.element_to_be_clickable((By.XPATH, xp))
#                     )
#                     btn.click()
#                     time.sleep(0.5)
#                     self.cookies_handled = True
#                     self.logger.info("Cookie popup dismissed.")
#                     return
#                 except Exception:
#                     continue
#         except Exception:
#             pass  # silent best-effort

#     def wait_for_page_load(self):
#         """Wait for page to fully load"""
#         try:
#             # Wait for body to be present
#             WebDriverWait(self.driver, 15).until(
#                 EC.presence_of_element_located((By.TAG_NAME, "body"))
#             )
#             # Additional wait for dynamic content
#             time.sleep(2)

#             # Try to wait for common loading indicators to disappear
#             loading_selectors = [
#                 '.loading', '.loader', '.spinner', '#loading',
#                 '[class*="loading"]', '[class*="spinner"]'
#             ]

#             for selector in loading_selectors:
#                 try:
#                     WebDriverWait(self.driver, 3).until(
#                         EC.invisibility_of_element_located((By.CSS_SELECTOR, selector))
#                     )
#                 except TimeoutException:
#                     pass  # Loading indicator might not be present

#             # NEW: try close cookie banner once
#             self._handle_cookie_popup_once()

#         except TimeoutException:
#             self.logger.warning("Page load timeout, proceeding anyway")
#         except Exception as e:
#             self.logger.error(f"Error waiting for page load: {str(e)}")

#     def _detect_language(self, text):
#         if not text:
#             return "unknown"
#         # prefer HTML lang if accurate
#         try:
#             lang = self.driver.execute_script("return document.documentElement.lang || document.documentElement.getAttribute('lang');")
#             if lang:
#                 return lang.split('-')[0].lower()
#         except Exception:
#             pass
#         if _lang_detect:
#             try:
#                 sample = text if len(text) < 1000 else text[:1000]
#                 return _lang_detect(sample)
#             except Exception:
#                 return "unknown"
#         return "unknown"

#     def extract_content_selenium(self, url):
#         """Extract content using Selenium"""
#         try:
#             self.driver.get(url)
#             self.wait_for_page_load()

#             # Get page source and parse with BeautifulSoup
#             soup = BeautifulSoup(self.driver.page_source, 'html.parser')

#             # Extract title
#             title = ""
#             title_elements = soup.find_all(['h1', 'title'])
#             for element in title_elements:
#                 if element and element.get_text().strip():
#                     title = self.clean_text(element.get_text())
#                     break

#             if not title and soup.title:
#                 title = self.clean_text(soup.title.get_text())

#             # Remove unwanted elements
#             unwanted_selectors = [
#                 'nav', 'header', 'footer', '.navigation', '.nav', '.menu',
#                 '.sidebar', '.ad', '.advertisement', '.social', '.share',
#                 '.comment', '.breadcrumb', 'script', 'style', '.skip-link',
#                 '.header', '.footer', '[role="navigation"]', '.site-header',
#                 '.site-footer', '.main-navigation', '.secondary-navigation',
#                 '#cookie', '.cookie', '[aria-label*="cookie"]'
#             ]

#             for selector in unwanted_selectors:
#                 for element in soup.select(selector):
#                     element.decompose()

#             # Extract main content
#             content_text = ""

#             # Try to find main content container
#             main_selectors = [
#                 'main', '.main-content', '#main-content', '.content',
#                 '.page-content', '.article-content', '.post-content',
#                 'article', '.main', '#content', '.entry-content',
#                 '.single-content', '.page-wrapper', '.content-wrapper'
#             ]

#             main_content = None
#             for selector in main_selectors:
#                 main_content = soup.select_one(selector)
#                 if main_content:
#                     break

#             # Fallback to body if no main content found
#             if not main_content:
#                 main_content = soup.find('body')

#             if main_content:
#                 # Extract structured content
#                 parts = []
#                 for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6',
#                                                       'p', 'div', 'span', 'li', 'td']):
#                     text = self.clean_text(element.get_text())
#                     if not text or len(text) < 3:
#                         continue
#                     parts.append(text)
#                 content_text = ' '.join(parts)

#             # NEW: language detection
#             language = self._detect_language(content_text)

#             return {
#                 "title": title if title else url.split('/')[-1],
#                 "content": content_text,
#                 "language": language,
#             }

#         except Exception as e:
#             self.logger.error(f"Error extracting content from {url}: {str(e)}")
#             return {"title": "", "content": "", "language": "unknown"}

#     def extract_links_selenium(self, url):
#         """Extract links using Selenium"""
#         links = set()
#         try:
#             # Find all links on the page
#             link_elements = self.driver.find_elements(By.TAG_NAME, "a")

#             for element in link_elements:
#                 try:
#                     href = element.get_attribute("href")
#                     if not href:
#                         continue

#                     # Convert to absolute URL
#                     absolute_url = urljoin(url, href)
#                     clean_url = absolute_url.split('#')[0]

#                     if not self.is_valid_url(clean_url):
#                         continue
#                     if not self._allowed_by_robots(clean_url):
#                         continue

#                     if clean_url not in self.visited_urls and clean_url not in self.enqueued_urls:
#                         links.add(clean_url)

#                     # Check for PDFs
#                     if href.lower().endswith('.pdf'):
#                         pdf_url = urljoin(url, href)
#                         if pdf_url not in self.pdf_links:
#                             self.pdf_links.append(pdf_url)

#                 except Exception:
#                     continue

#         except Exception as e:
#             self.logger.error(f"Error extracting links from {url}: {str(e)}")

#         return links

#     def crawl_page(self, url):
#         """Crawl a single page using Selenium"""
#         try:
#             self.logger.info(f"Crawling: {url}")

#             # Extract content
#             content_data = self.extract_content_selenium(url)

#             # Only save if we have meaningful content
#             if content_data["content"] and len(content_data["content"]) > 50:
#                 page_data = {
#                     "title": content_data["title"],
#                     "url": url,
#                     "content": content_data["content"],
#                     "language": content_data.get("language", "unknown")
#                 }
#                 self.extracted_data.append(page_data)
#                 self.logger.info(f"✓ Content extracted from: {url} ({len(content_data['content'])} chars)")
#             else:
#                 self.logger.warning(f"✗ No meaningful content found: {url}")

#             # Extract links for further crawling
#             new_links = self.extract_links_selenium(url)
#             add_count = 0
#             for link in new_links:
#                 if link not in self.visited_urls and link not in self.enqueued_urls:
#                     self.url_queue.append(link)
#                     self.enqueued_urls.add(link)
#                     add_count += 1
#             if add_count:
#                 self.logger.debug(f"Enqueued {add_count} new URLs from: {url}")

#             return True

#         except Exception as e:
#             self.logger.error(f"Error crawling {url}: {str(e)}")
#             return False

#     def extract_pdf_content(self, pdf_url):
#         """Extract text from PDF documents"""
#         try:
#             self.logger.info(f"Processing PDF: {pdf_url}")

#             response = self.session.get(pdf_url, timeout=30)
#             response.raise_for_status()

#             pdf_file = BytesIO(response.content)
#             pdf_reader = PyPDF2.PdfReader(pdf_file)

#             text_content = ""
#             for page in pdf_reader.pages:
#                 try:
#                     text_content += (page.extract_text() or "") + "\n\n"
#                 except Exception:
#                     continue

#             if text_content.strip():
#                 pdf_data = {
#                     "title": os.path.basename(pdf_url).replace('.pdf', '').replace('-', ' ').replace('_', ' ').title(),
#                     "url": pdf_url,
#                     "content": self.clean_text(text_content),
#                     "type": "PDF",
#                     "language": self._detect_language(text_content)
#                 }
#                 self.extracted_data.append(pdf_data)
#                 self.logger.info(f"✓ PDF content extracted: {pdf_url}")

#         except Exception as e:
#             self.logger.error(f"Error processing PDF {pdf_url}: {str(e)}")

#     def save_data(self, filename="meity_data.json"):
#         """Save extracted data to JSON file"""
#         try:
#             # Filter out empty content
#             filtered_data = [item for item in self.extracted_data if item.get('content', '').strip()]

#             with open(filename, 'w', encoding='utf-8') as f:
#                 json.dump(filtered_data, f, indent=2, ensure_ascii=False)

#             self.logger.info(f"Data saved to {filename}")
#             self.logger.info(f"Total valid entries: {len(filtered_data)}")

#         except Exception as e:
#             self.logger.error(f"Error saving data: {str(e)}")

#     def crawl(self, process_pdfs=True):
#         """Main crawling method"""
#         self.logger.info("Starting MeitY website crawl with Selenium...")

#         crawled_count = 0

#         try:
#             while self.url_queue and crawled_count < self.max_pages:
#                 url = self.url_queue.popleft()

#                 if url in self.visited_urls:
#                     continue

#                 # robots check again just in case
#                 if not self._allowed_by_robots(url):
#                     self.logger.info(f"Skipped by robots.txt: {url}")
#                     continue

#                 self.visited_urls.add(url)

#                 # Crawl the page
#                 success = self.crawl_page(url)

#                 if success:
#                     crawled_count += 1

#                 # Respectful delay
#                 time.sleep(self.delay)

#                 # Progress update
#                 if crawled_count % 5 == 0:
#                     self.logger.info(f"Progress: {crawled_count} pages crawled, {len(self.url_queue)} in queue, {len(self.extracted_data)} with content")

#             self.logger.info(f"Web crawling completed. Crawled {crawled_count} pages.")

#             # Process PDFs if requested
#             if process_pdfs and self.pdf_links:
#                 self.logger.info(f"Processing {len(self.pdf_links)} PDF documents...")
#                 for pdf_url in self.pdf_links[:50]:  # Limit PDFs for safety; adjust as needed
#                     self.extract_pdf_content(pdf_url)
#                     time.sleep(0.5)

#             # Save data
#             self.save_data()

#             # NEW: summary print with language distribution
#             langs = [d.get("language", "unknown") for d in self.extracted_data]
#             lang_counts = dict(Counter(langs))
#             html_pages = [d for d in self.extracted_data if d.get("type") != "PDF"]
#             pdf_pages = [d for d in self.extracted_data if d.get("type") == "PDF"]

#             print(f"\n{'='*60}")
#             print("🎉 CRAWLING COMPLETED SUCCESSFULLY!")
#             print(f"{'='*60}")
#             print(f"📄 Total HTML pages with content: {len(html_pages)}")
#             print(f"📋 PDF documents processed: {len(pdf_pages)}")
#             print(f"💬 Languages distribution: {lang_counts}")
#             print(f"💾 Data saved to: meity_data.json")
#             print(f"📊 Metadata saved to: meity_data/metadata.json" if os.path.exists('meity_data/metadata.json') else "")
#             print(f"{'='*60}")

#             # sample
#             if self.extracted_data:
#                 print("\n📝 Sample extracted content:")
#                 print("-" * 40)
#                 for item in self.extracted_data[:3]:
#                     print(f"Title: {item.get('title')}")
#                     print(f"URL: {item.get('url')}")
#                     print(f"Language: {item.get('language','unknown')}")
#                     preview = item.get('content', '')[:200].replace("\n", " ")
#                     print(f"Content preview: {preview}...")
#                     print("-" * 40)

#             return self.extracted_data

#         finally:
#             if self.driver:
#                 self.driver.quit()
#                 self.logger.info("WebDriver closed")


# def main():
#     """Main function to run the crawler"""

#     # Create output directory
#     if not os.path.exists('meity_data'):
#         os.makedirs('meity_data')

#     # Initialize crawler
#     crawler = MeitYSeleniumCrawler(
#         base_url="https://www.meity.gov.in",
#         delay=1,           # faster but still polite
#         max_pages=1000,    # increase to crawl more
#         headless=True,     # headless browser
#         respect_robots=True,  # follow robots.txt
#         use_sitemap=True      # seed from sitemap for completeness
#     )

#     # Start crawling
#     try:
#         data = crawler.crawl(process_pdfs=True)

#         # Save additional metadata
#         metadata = {
#             "total_pages": len([d for d in data if d.get("type") != "PDF"]),
#             "total_pdfs": len([item for item in data if item.get("type") == "PDF"]),
#             "crawl_date": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "base_url": crawler.base_url,
#             "method": "Selenium WebDriver",
#             "unique_urls_crawled": len(crawler.visited_urls)
#         }

#         with open("meity_data/metadata.json", 'w', encoding='utf-8') as f:
#             json.dump(metadata, f, indent=2, ensure_ascii=False)

#         print(f"\n{'='*60}")
#         print("✅ Metadata saved.")
#         print(f"Unique URLs visited: {metadata['unique_urls_crawled']}")
#         print(f"{'='*60}")

#     except KeyboardInterrupt:
#         print("\n⚠️  Crawling interrupted by user.")
#         crawler.save_data("meity_data_partial.json")
#         print("💾 Partial data saved to: meity_data_partial.json")
#     except Exception as e:
#         print(f"❌ Error during crawling: {str(e)}")
#         if crawler.extracted_data:
#             crawler.save_data("meity_data_error.json")
#             print("💾 Partial data saved to: meity_data_error.json")
#     finally:
#         if crawler.driver:
#             crawler.driver.quit()


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
Enhanced MeitY Website Crawler - Complete Data Extraction
Addresses issues with content summarization and "no meaningful content"
Captures complete data from tables, forms, and all content types.
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

from bs4 import BeautifulSoup
import json
import time
import os
import re
from urllib.parse import urljoin, urlparse, unquote
from collections import deque
import logging
import requests
import PyPDF2
from io import BytesIO
import xml.etree.ElementTree as ET

# Extra imports for improvements
from concurrent.futures import ThreadPoolExecutor, as_completed
from langdetect import detect, DetectorFactory
import urllib.robotparser

# Try optional pdfminer for better PDF text in some cases
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    PDFMINER_AVAILABLE = True
except Exception:
    PDFMINER_AVAILABLE = False

# Make langdetect deterministic
DetectorFactory.seed = 0

class EnhancedMeitYCrawler:
    def __init__(self, base_url="https://www.meity.gov.in", delay=2, max_pages=1000, headless=True,
                 respect_robots=True, parallel_pdfs=True):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.delay = delay
        self.max_pages = max_pages
        self.parallel_pdfs = parallel_pdfs
        self.respect_robots = respect_robots
        
        # Setup Chrome options
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument('--headless=new')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--window-size=1920,1080')
        self.chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        self.chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.chrome_options.add_experimental_option('useAutomationExtension', False)
        self.chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Initialize WebDriver
        self.driver = None
        
        # Data storage
        self.visited_urls = set()
        self.extracted_data = []
        self.pdf_links = set()
        self.url_queue = deque([base_url])
        self.enqueued_urls = set([base_url])
        
        # Minimal cookie patterns for removal (only obvious cookie banners)
        self.cookie_patterns = [
            r'this website uses cookies.*?accept all',
            r'we use cookies.*?accept',
            r'cookies? settings?.*?accept',
        ]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.setup_driver()
        
        # Setup requests session for PDFs and sitemaps
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Robots.txt handling
        self.robots_parser = None
        if self.respect_robots:
            try:
                robots_url = urljoin(self.base_url, "/robots.txt")
                self.robots_parser = urllib.robotparser.RobotFileParser()
                self.robots_parser.set_url(robots_url)
                self.robots_parser.read()
                self.logger.info(f"Loaded robots.txt from {robots_url}")
            except Exception as e:
                self.logger.warning(f"Could not read robots.txt: {str(e)}")
                self.robots_parser = None

    def setup_driver(self):
        """Setup Chrome WebDriver"""
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=self.chrome_options)
            self.driver.set_page_load_timeout(30)
            self.driver.implicitly_wait(10)
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.logger.info("WebDriver setup successful")
        except Exception as e:
            self.logger.error(f"Error setting up WebDriver: {str(e)}")
            raise
    
    def handle_cookie_consent(self):
        """Handle cookie consent pop-ups - minimal removal"""
        try:
            # Common cookie consent selectors - only most obvious ones
            consent_selectors = [
                "button[id*='accept']",
                "button[class*='accept']",
                ".cookie-accept",
                "#accept-cookies"
            ]
            
            for selector in consent_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        elements[0].click()
                        time.sleep(2)
                        self.logger.info("Cookie consent handled")
                        return True
                except Exception:
                    continue
                    
        except Exception as e:
            self.logger.debug(f"Cookie consent handling failed: {str(e)}")
        
        return False
    
    def normalize_url(self, url):
        """Normalize URL for better deduplication"""
        try:
            parsed = urlparse(url)
            # Remove fragments and common tracking parameters
            query_parts = []
            if parsed.query:
                for part in parsed.query.split('&'):
                    if not any(param in part.lower() for param in ['utm_', 'fbclid', 'gclid', '_ga', 'ref']):
                        query_parts.append(part)
            
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if query_parts:
                normalized += f"?{'&'.join(query_parts)}"
                
            return normalized.rstrip('/')
        except Exception:
            return url
    
    def is_valid_url(self, url):
        """Check if URL is valid, internal, and allowed by robots"""
        try:
            parsed = urlparse(url)
            valid_domain = (
                parsed.netloc == self.domain or 
                parsed.netloc == f"www.{self.domain}" or
                parsed.netloc == self.domain.replace("www.", "")
            )
            if not valid_domain:
                return False
            
            # Skip certain file types and admin URLs
            skip_patterns = [
                r'\.(css|js|ico|png|jpg|jpeg|gif|svg|woff|woff2|ttf)$',
                r'/admin/', r'/wp-admin/', r'/wp-content/',
                r'#', r'mailto:', r'tel:', r'javascript:'
            ]
            
            for pattern in skip_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return False
            
            # Robots.txt check
            if self.robots_parser and not self.robots_parser.can_fetch("*", url):
                self.logger.debug(f"Blocked by robots.txt: {url}")
                return False
                
            return True
        except Exception:
            return False
    
    def minimal_text_cleaning(self, text):
        """Minimal text cleaning - preserve all meaningful content"""
        if not text:
            return ""
        
        # Only remove obvious cookie consent text using patterns
        for pattern in self.cookie_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Minimal whitespace normalization - preserve structure
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        
        return text.strip()

    def detect_language(self, text):
        try:
            if text and len(text) > 20:
                return detect(text)
        except Exception:
            pass
        return "unknown"
    
    def wait_for_page_load(self):
        """Enhanced page loading wait"""
        try:
            # Wait for body to load
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Handle cookie consent first
            self.handle_cookie_consent()
            
            # Wait a bit more for dynamic content
            time.sleep(3)
            
            # Wait for common loading elements to disappear
            loading_selectors = [
                '.loading', '.loader', '.spinner', '#loading', 
                '[class*="loading"]', '[class*="spinner"]',
                '.preloader', '#preloader'
            ]
            
            for selector in loading_selectors:
                try:
                    WebDriverWait(self.driver, 5).until(
                        EC.invisibility_of_element_located((By.CSS_SELECTOR, selector))
                    )
                except TimeoutException:
                    pass
                    
        except TimeoutException:
            self.logger.warning("Page load timeout, proceeding anyway")
        except Exception as e:
            self.logger.error(f"Error waiting for page load: {str(e)}")
    
    def extract_complete_content(self, url):
        """Complete content extraction - no summarization, capture everything"""
        try:
            self.driver.get(url)
            self.wait_for_page_load()
            
            # Get page source and create soup
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Extract title with multiple strategies
            title = ""
            title_selectors = [
                'h1', '.page-title', '.main-title', '.content-title',
                '.entry-title', '.post-title', '.article-title', 'title'
            ]
            
            for selector in title_selectors:
                elements = soup.select(selector)
                for element in elements:
                    if element and element.get_text().strip():
                        title = self.minimal_text_cleaning(element.get_text())
                        break
                if title:
                    break
            
            if not title and soup.title:
                title = self.minimal_text_cleaning(soup.title.get_text())
            
            # Remove only essential unwanted elements (keep most content)
            essential_unwanted = [
                'script', 'style', 'noscript',
                '.cookie-consent', '.cookie-banner', '.cookie-notice',
                '[style*="display: none"]', '[style*="visibility: hidden"]'
            ]
            
            for selector in essential_unwanted:
                for element in soup.select(selector):
                    element.decompose()
            
            # Extract ALL content including tables, forms, lists
            all_content = []
            
            # Get main content area first
            main_selectors = [
                'main', '.main-content', '#main-content', '.content',
                '.page-content', '.article-content', '.post-content',
                'article', '.main', '#content', '.entry-content',
                '.single-content', '.page-wrapper', '.content-wrapper',
                '.container', '.wrapper'
            ]
            
            main_content = None
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.find('body')
            
            if main_content:
                # Extract headings
                for heading in main_content.find_all(['h1','h2','h3','h4','h5','h6']):
                    if heading.get_text().strip():
                        level = heading.name
                        text = self.minimal_text_cleaning(heading.get_text())
                        if text:
                            all_content.append(f"\n{'#' * int(level[1:])} {text}\n")
                
                # Extract paragraphs
                for para in main_content.find_all('p'):
                    text = self.minimal_text_cleaning(para.get_text())
                    if text and len(text) > 10:  # Very minimal threshold
                        all_content.append(f"{text}\n")
                
                # Extract tables - COMPLETE table data
                for table in main_content.find_all('table'):
                    table_data = []
                    
                    # Extract table headers
                    headers = []
                    for th in table.find_all('th'):
                        headers.append(self.minimal_text_cleaning(th.get_text()))
                    
                    if headers:
                        table_data.append("| " + " | ".join(headers) + " |")
                        table_data.append("|" + "---|" * len(headers))
                    
                    # Extract table rows
                    for row in table.find_all('tr'):
                        cells = []
                        for cell in row.find_all(['td', 'th']):
                            cell_text = self.minimal_text_cleaning(cell.get_text())
                            cells.append(cell_text if cell_text else " ")
                        
                        if cells and any(cell.strip() for cell in cells):
                            table_data.append("| " + " | ".join(cells) + " |")
                    
                    if table_data:
                        all_content.append("\n**TABLE:**\n")
                        all_content.extend(table_data)
                        all_content.append("\n")
                
                # Extract lists - COMPLETE list data
                for ul_ol in main_content.find_all(['ul', 'ol']):
                    list_items = []
                    for li in ul_ol.find_all('li', recursive=False):
                        text = self.minimal_text_cleaning(li.get_text())
                        if text:
                            list_items.append(f"• {text}")
                    
                    if list_items:
                        all_content.extend(list_items)
                        all_content.append("\n")
                
                # Extract forms - COMPLETE form data
                for form in main_content.find_all('form'):
                    form_data = ["\n**FORM:**\n"]
                    
                    # Form title or legend
                    legend = form.find('legend')
                    if legend:
                        form_data.append(f"Form: {self.minimal_text_cleaning(legend.get_text())}\n")
                    
                    # Form fields
                    for field in form.find_all(['input', 'select', 'textarea', 'label']):
                        if field.name == 'label':
                            text = self.minimal_text_cleaning(field.get_text())
                            if text:
                                form_data.append(f"Label: {text}")
                        else:
                            field_type = field.get('type', field.name)
                            placeholder = field.get('placeholder', '')
                            name = field.get('name', '')
                            value = field.get('value', '')
                            
                            field_info = f"Field: {field_type}"
                            if name:
                                field_info += f" (name: {name})"
                            if placeholder:
                                field_info += f" (placeholder: {placeholder})"
                            if value and field_type not in ['password']:
                                field_info += f" (value: {value})"
                            
                            form_data.append(field_info)
                    
                    if len(form_data) > 1:
                        all_content.extend(form_data)
                        all_content.append("\n")
                
                # Extract divs with significant text content
                for div in main_content.find_all('div'):
                    # Skip if div contains other structural elements
                    if div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table', 'ul', 'ol', 'form']):
                        continue
                    
                    text = self.minimal_text_cleaning(div.get_text())
                    if text and len(text) > 30:  # Only substantial div content
                        all_content.append(f"{text}\n")
                
                # Extract spans with significant content
                for span in main_content.find_all('span'):
                    # Skip if span is inside other elements we've already processed
                    if span.find_parent(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th']):
                        continue
                    
                    text = self.minimal_text_cleaning(span.get_text())
                    if text and len(text) > 20:
                        all_content.append(f"{text}\n")
                
                # Extract any remaining text content
                remaining_text = self.minimal_text_cleaning(main_content.get_text())
                current_content = ''.join(all_content)
                
                # Add any substantial text not already captured
                lines = remaining_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if len(line) > 15 and line not in current_content:
                        all_content.append(f"{line}\n")
            
            final_content = ''.join(all_content)
            
            return {
                "title": title if title else url.split('/')[-1],
                "content": final_content
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            return {"title": "", "content": ""}
    
    def extract_links_selenium(self, url):
        """Enhanced link extraction using multiple strategies"""
        links = set()
        try:
            # Extract regular links
            link_elements = self.driver.find_elements(By.TAG_NAME, "a")
            for element in link_elements:
                try:
                    href = element.get_attribute("href")
                    if not href:
                        continue
                        
                    absolute_url = urljoin(url, href)
                    clean_url = self.normalize_url(absolute_url)
                    
                    if self.is_valid_url(clean_url) and clean_url not in self.visited_urls and clean_url not in self.enqueued_urls:
                        links.add(clean_url)
                    
                    # Check for PDF links
                    if href.lower().endswith('.pdf') or 'pdf' in href.lower():
                        pdf_url = urljoin(url, href)
                        self.pdf_links.add(pdf_url)
                        
                except Exception:
                    continue
            
            # Look for links in JavaScript or dynamic content
            try:
                # Execute JavaScript to find more links
                js_links = self.driver.execute_script("""
                    var links = [];
                    var elements = document.querySelectorAll('[href], [data-href], [data-url]');
                    for (var i = 0; i < elements.length; i++) {
                        var href = elements[i].getAttribute('href') || 
                                  elements[i].getAttribute('data-href') || 
                                  elements[i].getAttribute('data-url');
                        if (href) links.push(href);
                    }
                    return links;
                """)
                
                for href in js_links:
                    if href:
                        absolute_url = urljoin(url, href)
                        clean_url = self.normalize_url(absolute_url)
                        
                        if self.is_valid_url(clean_url) and clean_url not in self.visited_urls and clean_url not in self.enqueued_urls:
                            links.add(clean_url)
                            
            except Exception as e:
                self.logger.debug(f"JavaScript link extraction failed: {str(e)}")
            
            # Look for PDFs in page content
            try:
                page_text = self.driver.page_source
                pdf_pattern = r'href=["\']([^"\']*\.pdf[^"\']*)["\']'
                pdf_matches = re.findall(pdf_pattern, page_text, re.IGNORECASE)
                
                for pdf_match in pdf_matches:
                    pdf_url = urljoin(url, pdf_match)
                    self.pdf_links.add(pdf_url)
                    
            except Exception:
                pass
            
        except Exception as e:
            self.logger.error(f"Error extracting links from {url}: {str(e)}")
        
        return links
    
    def crawl_page(self, url):
        """Enhanced page crawling with complete content extraction"""
        try:
            self.logger.info(f"Crawling: {url}")
            
            content_data = self.extract_complete_content(url)
            
            # Much more lenient content threshold - accept almost anything with text
            if content_data["content"] and len(content_data["content"].strip()) > 10:
                page_data = {
                    "title": content_data["title"],
                    "url": url,
                    "content": content_data["content"],
                    "language": self.detect_language(content_data["content"]),
                    "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.extracted_data.append(page_data)
                self.logger.info(f"✓ Content extracted from: {url} ({len(content_data['content'])} chars)")
            else:
                # Log what was found for debugging
                self.logger.warning(f"✗ Minimal content found: {url} ({len(content_data.get('content', ''))} chars)")
                # Still save it if there's any content at all
                if content_data["content"]:
                    page_data = {
                        "title": content_data["title"],
                        "url": url,
                        "content": content_data["content"],
                        "language": self.detect_language(content_data["content"]) if content_data["content"] else "unknown",
                        "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "note": "minimal_content"
                    }
                    self.extracted_data.append(page_data)
            
            # Extract new links
            new_links = self.extract_links_selenium(url)
            added_count = 0
            
            for link in new_links:
                if link not in self.visited_urls and link not in self.enqueued_urls:
                    self.url_queue.append(link)
                    self.enqueued_urls.add(link)
                    added_count += 1
            
            if added_count > 0:
                self.logger.info(f"Added {added_count} new URLs to queue")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {str(e)}")
            return False
    
    def discover_links_from_sitemap(self):
        """Discover URLs from sitemap"""
        try:
            sitemap_urls = [
                urljoin(self.base_url, "/sitemap.xml"),
                urljoin(self.base_url, "/sitemap_index.xml"),
                urljoin(self.base_url, "/sitemaps/sitemap.xml"),
            ]
            
            for sitemap_url in sitemap_urls:
                try:
                    response = self.session.get(sitemap_url, timeout=10)
                    if response.status_code == 200:
                        root = ET.fromstring(response.content)
                        
                        # Handle sitemap index
                        for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                            loc = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                            if loc is not None:
                                self.discover_links_from_sitemap_url(loc.text)
                        
                        # Handle direct sitemap
                        for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                            loc = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                            if loc is not None:
                                clean_url = self.normalize_url(loc.text)
                                if self.is_valid_url(clean_url) and clean_url not in self.enqueued_urls:
                                    self.url_queue.append(clean_url)
                                    self.enqueued_urls.add(clean_url)
                        
                        self.logger.info(f"Discovered URLs from sitemap: {sitemap_url}")
                        return True
                        
                except Exception as e:
                    self.logger.debug(f"Could not parse sitemap {sitemap_url}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Error discovering sitemap URLs: {str(e)}")
        
        return False
    
    def discover_links_from_sitemap_url(self, sitemap_url):
        """Parse individual sitemap URL"""
        try:
            response = self.session.get(sitemap_url, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                    loc = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc is not None:
                        clean_url = self.normalize_url(loc.text)
                        if self.is_valid_url(clean_url) and clean_url not in self.enqueued_urls:
                            self.url_queue.append(clean_url)
                            self.enqueued_urls.add(clean_url)
        except Exception as e:
            self.logger.debug(f"Error parsing sitemap {sitemap_url}: {str(e)}")
    
    def extract_pdf_content(self, pdf_url):
        """Enhanced PDF content extraction"""
        try:
            self.logger.info(f"Processing PDF: {pdf_url}")
            
            response = self.session.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            pdf_file = BytesIO(response.content)
            text_content = ""
            
            # Try PyPDF2 first
            try:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            text_content += page_text + "\n\n"
                    except Exception as e:
                        self.logger.debug(f"Error extracting page {page_num} from {pdf_url}: {str(e)}")
                        continue
            except Exception as e:
                self.logger.warning(f"PyPDF2 failed on {pdf_url}: {e}")
            
            # Fallback to pdfminer if available and PyPDF2 failed
            if not text_content.strip() and PDFMINER_AVAILABLE:
                try:
                    pdf_file.seek(0)
                    text_content = pdfminer_extract_text(pdf_file)
                    self.logger.info(f"Used pdfminer for {pdf_url}")
                except Exception as e:
                    self.logger.warning(f"pdfminer also failed on {pdf_url}: {e}")
            
            if text_content.strip():
                clean_content = self.minimal_text_cleaning(text_content)
                if len(clean_content) > 20:  # Very minimal threshold for PDFs
                    lang = self.detect_language(clean_content)
                    pdf_data = {
                        "title": os.path.basename(pdf_url).replace('.pdf', '').replace('-', ' ').replace('_', ' ').title(),
                        "url": pdf_url,
                        "content": clean_content,
                        "type": "PDF",
                        "language": lang,
                        "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    self.logger.info(f"✓ PDF content extracted: {pdf_url} ({len(clean_content)} chars)")
                    return pdf_data
                    
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_url}: {str(e)}")
        
        return None
    
    def process_pdfs_parallel(self, max_workers=3, limit=50):
        """Enhanced parallel PDF processing"""
        pdf_list = list(self.pdf_links)
        to_process = pdf_list[:limit] if limit else pdf_list
        
        self.logger.info(f"Processing {len(to_process)} PDF documents in parallel...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.extract_pdf_content, pdf_url): pdf_url for pdf_url in to_process}
            
            for future in as_completed(futures):
                try:
                    data = future.result()
                    if data:
                        self.extracted_data.append(data)
                except Exception as e:
                    pdf_url = futures[future]
                    self.logger.error(f"Error processing PDF {pdf_url}: {str(e)}")
        
        self.logger.info("✓ Parallel PDF processing completed")

    def save_data(self, filename="meity_complete_data.json"):
        """Save data with minimal filtering - preserve everything"""
        try:
            # Very minimal filtering - only remove truly empty entries
            filtered_data = []
            seen_urls = set()
            
            for item in self.extracted_data:
                content = item.get('content', '').strip()
                url = item.get('url', '')
                
                # Only filter out completely empty content or duplicate URLs
                if content and url not in seen_urls:
                    seen_urls.add(url)
                    filtered_data.append(item)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Complete data saved to {filename}")
            self.logger.info(f"Total entries saved: {len(filtered_data)}")
            
            return len(filtered_data)
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            return 0
    
    def crawl(self, process_pdfs=True):
        """Enhanced main crawling method with complete data preservation"""
        self.logger.info("Starting comprehensive MeitY website crawl with complete data extraction...")
        
        # First, try to discover URLs from sitemap
        self.discover_links_from_sitemap()
        
        crawled_count = 0
        successful_extractions = 0
        
        try:
            while self.url_queue and crawled_count < self.max_pages:
                url = self.url_queue.popleft()
                
                if url in self.visited_urls:
                    continue
                
                self.visited_urls.add(url)
                before_count = len(self.extracted_data)
                
                success = self.crawl_page(url)
                
                if success:
                    crawled_count += 1
                    
                    # Check if we actually extracted content
                    if len(self.extracted_data) > before_count:
                        successful_extractions += 1
                
                time.sleep(self.delay)
                
                # Progress reporting - more detailed
                if crawled_count % 10 == 0:
                    self.logger.info(
                        f"Progress: {crawled_count} pages crawled, "
                        f"{successful_extractions} with content extracted, "
                        f"{len(self.url_queue)} in queue, "
                        f"{len(self.pdf_links)} PDFs found"
                    )
            
            self.logger.info(f"Web crawling completed. Crawled {crawled_count} pages, extracted content from {successful_extractions} pages, found {len(self.pdf_links)} PDFs")
            
            # Process PDFs
            if process_pdfs and self.pdf_links:
                if self.parallel_pdfs:
                    self.process_pdfs_parallel(max_workers=3, limit=100)
                else:
                    for pdf_url in list(self.pdf_links)[:50]:
                        pdf_data = self.extract_pdf_content(pdf_url)
                        if pdf_data:
                            self.extracted_data.append(pdf_data)
                        time.sleep(1)
            
            # Save data
            total_saved = self.save_data()
            return self.extracted_data
            
        except KeyboardInterrupt:
            self.logger.info("Crawling interrupted by user")
            self.save_data("meity_complete_data_partial.json")
            return self.extracted_data
        finally:
            if self.driver:
                self.driver.quit()
                self.logger.info("WebDriver closed")

def main():
    """Enhanced main function with complete data extraction"""
    if not os.path.exists('meity_complete_data'):
        os.makedirs('meity_complete_data')
    
    # Initialize crawler with enhanced settings for complete data extraction
    crawler = EnhancedMeitYCrawler(
        base_url="https://www.meity.gov.in",
        delay=2,  # Respectful delay
        max_pages=1000,  # Comprehensive crawling
        headless=True,  # Set to False for debugging
        respect_robots=True,
        parallel_pdfs=True
    )
    
    try:
        data = crawler.crawl(process_pdfs=True)
        
        # Enhanced metadata collection
        lang_counts = {}
        type_counts = {"HTML": 0, "PDF": 0}
        total_content_length = 0
        minimal_content_count = 0
        
        for item in data:
            lang = item.get("language", "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            
            item_type = item.get("type", "HTML")
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
            
            content_length = len(item.get("content", ""))
            total_content_length += content_length
            
            # Track minimal content items
            if item.get("note") == "minimal_content":
                minimal_content_count += 1
        
        metadata = {
            "total_pages": type_counts["HTML"],
            "total_pdfs": type_counts["PDF"],
            "total_entries": len(data),
            "successful_extractions": len(data) - minimal_content_count,
            "minimal_content_pages": minimal_content_count,
            "total_content_chars": total_content_length,
            "average_content_length": total_content_length // len(data) if data else 0,
            "crawl_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_url": crawler.base_url,
            "method": "Enhanced Selenium WebDriver - Complete Data Extraction",
            "languages": lang_counts,
            "type_distribution": type_counts,
            "total_urls_visited": len(crawler.visited_urls),
            "total_urls_found": len(crawler.enqueued_urls),
            "extraction_approach": "Complete data preservation with minimal summarization"
        }
        
        # Save metadata
        with open("meity_complete_data/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save URL lists for reference
        url_data = {
            "visited_urls": list(crawler.visited_urls),
            "pdf_urls": list(crawler.pdf_links),
            "total_discovered": len(crawler.enqueued_urls)
        }
        
        with open("meity_complete_data/urls_discovered.json", 'w', encoding='utf-8') as f:
            json.dump(url_data, f, indent=2, ensure_ascii=False)
        
        # Print comprehensive results
        print(f"\n{'='*80}")
        print("ENHANCED MeitY CRAWLING WITH COMPLETE DATA EXTRACTION COMPLETED!")
        print(f"{'='*80}")
        print(f"CRAWLING STATISTICS:")
        print(f"   HTML pages with content: {metadata['total_pages']}")
        print(f"   PDF documents processed: {metadata['total_pdfs']}")
        print(f"   Total content entries: {metadata['total_entries']}")
        print(f"   Successful content extractions: {metadata['successful_extractions']}")
        print(f"   Pages with minimal content: {metadata['minimal_content_pages']}")
        print(f"   Total URLs visited: {metadata['total_urls_visited']}")
        print(f"   Total URLs discovered: {metadata['total_urls_found']}")
        print(f"   Total content characters: {metadata['total_content_chars']:,}")
        print(f"   Average content length: {metadata['average_content_length']:,} chars")
        print()
        print(f"LANGUAGE DISTRIBUTION:")
        for lang, count in sorted(metadata['languages'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {lang}: {count} documents")
        print()
        print(f"FILES SAVED:")
        print(f"   Main data: meity_complete_data.json")
        print(f"   Metadata: meity_complete_data/metadata.json")
        print(f"   URL list: meity_complete_data/urls_discovered.json")
        print(f"{'='*80}")
        
        if data:
            print(f"\nSAMPLE EXTRACTED CONTENT:")
            print(f"{'-'*60}")
            
            # Show samples with complete data
            for i, item in enumerate(data[:4], 1):
                print(f"Sample {i} ({'PDF' if item.get('type') == 'PDF' else 'HTML'}):")
                print(f"  Title: {item['title'][:80]}...")
                print(f"  URL: {item['url']}")
                print(f"  Language: {item.get('language', 'unknown')}")
                print(f"  Content length: {len(item['content'])} characters")
                print(f"  Content type: {'Minimal' if item.get('note') == 'minimal_content' else 'Full extraction'}")
                
                # Show more content preview for debugging
                content_preview = item['content'][:400].replace('\n', ' ')
                print(f"  Content preview: {content_preview}...")
                print(f"{'-'*60}")
        
        # Additional debugging info for "no meaningful content" issues
        print(f"\nDEBUGGING INFO:")
        print(f"   Pages that had minimal content: {minimal_content_count}")
        print(f"   Success rate: {((len(data) - minimal_content_count) / len(data) * 100):.1f}% if data else 0")
        print(f"   Approach: Complete data extraction with minimal filtering")
        print(f"   Content threshold: Very low (10+ characters)")
        print(f"   Cleaning: Minimal (only cookie banners)")
        
        print(f"\nCrawling completed successfully!")
        print(f"Check the 'meity_complete_data' directory for all saved files.")
        
    except KeyboardInterrupt:
        print("\nCrawling interrupted by user.")
        if hasattr(crawler, 'extracted_data') and crawler.extracted_data:
            crawler.save_data("meity_complete_data_partial.json")
            print("Partial data saved to: meity_complete_data_partial.json")
    
    except Exception as e:
        print(f"Error during crawling: {str(e)}")
        if hasattr(crawler, 'extracted_data') and crawler.extracted_data:
            crawler.save_data("meity_complete_data_error.json")
            print("Partial data saved to: meity_complete_data_error.json")
    
    finally:
        if hasattr(crawler, 'driver') and crawler.driver:
            crawler.driver.quit()

if __name__ == "__main__":
    main()