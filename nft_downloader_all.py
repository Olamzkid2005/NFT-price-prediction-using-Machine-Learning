import os
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from urllib.parse import urljoin

# === CONFIGURATION ===
collection_slug = "doodles-official"  # Change this to your collection
base_url = f"https://opensea.io/collection/{collection_slug}"
save_dir = f"nft_images_{collection_slug}"
scroll_limit = 5  # Number of scrolls to load more NFTs

# === SETUP CHROME DRIVER (headless) ===
options = Options()
options.headless = True
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(options=options)

# Create directory if not exist
os.makedirs(save_dir, exist_ok=True)

def scroll_page(scrolls=5, delay=2):
    driver.get(base_url)
    time.sleep(5)  # Let page load initially
    for _ in range(scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)

def scrape_nfts():
    soup = BeautifulSoup(driver.page_source, "html.parser")
    nft_cards = soup.find_all("a", href=True)

    seen = set()
    count = 0
    for a in nft_cards:
        href = a['href']
        if "/assets/" in href and href not in seen:
            seen.add(href)
            img = a.find("img")
            if img and img.get("src"):
                name = a.get("title") or f"nft_{count}"
                img_url = img["src"]
                download_image(img_url, name)
                count += 1

def download_image(img_url, name):
    try:
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()
        ext = img_url.split(".")[-1].split("?")[0]
        name_clean = name.replace(" ", "_").replace("/", "").replace("#", "")
        file_path = os.path.join(save_dir, f"{name_clean}.{ext}")
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"[+] Downloaded {name_clean}")
    except Exception as e:
        print(f"[!] Failed to download {name}: {e}")

if __name__ == "__main__":
    print(f"🚀 Scraping NFT images from {base_url}")
    scroll_page(scroll_limit)
    scrape_nfts()
    driver.quit()
    print(f"\n✅ Done! Images saved in ./{save_dir}")
