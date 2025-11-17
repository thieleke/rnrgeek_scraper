#!/usr/bin/env python3
"""
download_episodes.py
--------------------
Complete podcast episode download workflow for the Rock and Roll Geek Show archive on podbean.com.

1. Download podcast list pages from Podbean
   - Downloads all HTML pages (page001.html, page002.html, etc.)
   - Saves to output_podbean/ directory

2. Extract episode URLs from HTML page files
   - Parses each downloaded HTML page
   - Extracts episode links from <div id='episode-list-content'>
   - Writes all URLs to episode_links.json

3. Download each episode's metadata, description, and audio file
   - Fetches the episode page and extracts JSON-LD metadata
   - Saves metadata.json with full episode data
   - Saves description.txt with episode description (or "No Description" if empty)
   - Downloads the media file using the original filename from contentUrl

All HTTP requests use a randomized User-Agent header from a predefined pool to avoid detection.

Episode folder structure:
    Episodes/<datePublished> - <episode-name>/
        metadata.json
        description.txt
        <original-media-filename>.mp3

The script automatically skips episodes that already exist with all required files (metadata, description, audio).

Configuration:
    - BASE_URL: Podbean podcast page URL
    - NUM_PAGES: Number of podcast list pages to download
    - PAGE_DELAY_IN_SEC: Delay between downloading list pages
    - DELAY_IN_SEC: Delay between downloading individual episodes
"""

import json
import logging
import os
import pathlib
import random
import re
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# Configure logging to both console and file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler('output.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Prevent propagation to root logger (avoid duplicate messages)
logger.propagate = False

# Podcast page download configuration
BASE_URL = "https://www.podbean.com/podcast-detail/nmr9z-42c7f/The-Rock-and-Roll-Geek-Show-Podcast?page="
NUM_PAGES = 263  # TODO: set this to the actual number of pages (accurate as of November 2025)
PAGE_DELAY_IN_SEC = 1  # delay between downloading podcast list pages

# Episode link extraction configuration
DEFAULT_INPUT_DIR = pathlib.Path("./output_podbean")
FILE_TEMPLATE = "page{num:03}.html"   # e.g. page001.html, page002.html

# Episode download configuration
INPUT_JSON   = pathlib.Path("episode_links.json")  # list of page URLs
OUTPUT_ROOT  = pathlib.Path("Episodes")            # top-level folder
DELAY_IN_SEC = 0.5                                 # pause between episode downloads
HTTP_TIMEOUT = 30                                  # timeout for HTTP requests in seconds

# User-Agent pool - randomize to avoid detection
USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Firefox on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Safari on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Firefox on Linux
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Opera on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0",
    # Brave on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Brave/120.0.0.0"
]


def get_random_user_agent() -> str:
    """Return a random user agent from the pool."""
    return random.choice(USER_AGENTS)


def download_podcast_pages(base_url: str, num_pages: int, output_dir: pathlib.Path, delay: float = 1.0) -> None:
    """
    Download all podcast list pages from Podbean.

    Args:
        base_url: Base URL with page parameter (e.g., "...?page=")
        num_pages: Number of pages to download
        output_dir: Directory to save downloaded HTML files
        delay: Seconds to wait between requests (default: 1.0)
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {num_pages} podcast list pages...")

    for page_num in range(1, num_pages + 1):
        file_path = output_dir / FILE_TEMPLATE.format(num=page_num)
        url = f"{base_url}{page_num}"

        logger.info(f"Downloading page {page_num}/{num_pages} -> {file_path.name}")

        try:
            headers = {"User-Agent": get_random_user_agent()}
            resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(resp.text)

        except Exception as exc:
            logger.error(f"Failed to download page {page_num}: {exc}")
            continue

        # Delay between requests (skip delay on last page)
        if page_num < num_pages and delay > 0:
            time.sleep(delay)

    logger.info(f"All {num_pages} pages downloaded to '{output_dir}'")


def extract_links_from_file(file_path: pathlib.Path) -> list[str]:
    """Return a list of URLs found inside <div id='episode-list-content'>."""
    if not file_path.is_file():
        logger.warning(f"Missing file: {file_path}")
        return []

    try:
        with file_path.open(encoding="utf-8") as fh:
            html = fh.read()
    except Exception as exc:
        logger.error(f"Could not read {file_path}: {exc}")
        return []

    soup = BeautifulSoup(html, "lxml")
    container = soup.find("div", id="episode-list-content")
    if not container:
        logger.warning(f'<div id="episode-list-content"> not found in {file_path}')
        return []

    # Grab every <a> inside that div which has a href attribute.
    return [a["href"] for a in container.find_all("a", href=True)]


def collect_episode_urls(input_dir: pathlib.Path, output_file: pathlib.Path) -> None:
    """Collect all episode URLs from HTML page files and write them to a JSON file."""
    all_links: list[str] = []
    page_num = 1

    while True:
        file_path = input_dir / FILE_TEMPLATE.format(num=page_num)

        # Stop if the page file doesn't exist
        if not file_path.is_file():
            logger.info(f"No more pages found (stopped at page{page_num:03}.html)")
            break

        page_links = extract_links_from_file(file_path)
        if page_links:
            logger.info(f"{len(page_links)} links collected from {file_path.name}")
            all_links.extend(page_links)

        page_num += 1

    logger.info(f"{len(all_links)} total links found across {page_num - 1} pages.")

    # Remove duplicates - same episode might appear on multiple pages
    unique_links = list(dict.fromkeys(all_links))  # Preserves order, removes duplicates
    if len(unique_links) < len(all_links):
        logger.info(f"Removed {len(all_links) - len(unique_links)} duplicate URLs")

    # Write output as JSON - a plain array
    try:
        with output_file.open("w", encoding="utf-8") as out:
            json.dump(unique_links, out, ensure_ascii=False, indent=2)
        logger.info(f"{len(unique_links)} unique links written to {output_file}")
    except Exception as exc:
        logger.error(f"Could not write to {output_file}: {exc}")


def sanitize_name(name: str, max_length: int = 250) -> str:
    """
    Replace every character that is unsafe for a file name with an
    underscore, keep letters, digits, hyphens, underscores and spaces.
    Collapse multiple spaces or underscores into a single space,
    and trim surrounding whitespace.
    Limit the result to max_length characters (default 250).
    """
    name = name.strip()
    name = re.sub(r'[^A-Za-z0-9_\- ]+', "_", name)
    name = re.sub(r"[_\s]+", " ", name)
    name = name or "episode"

    # Truncate to max_length if needed
    if len(name) > max_length:
        name = name[:max_length].rstrip()

    return name


def filename_from_contenturl(url: str, max_length: int = 250) -> str:
    """
    Given a contentUrl (e.g.
    https://traffic.libsyn.com/secure/rockandrollgeek/RnRGeekShow_1033.mp3?dest-id=115736)
    return a safe file name:  RnRGeekShow_1033.mp3
    Limit the total filename to max_length characters (default 250) while preserving the extension.
    """
    parsed = urlparse(url)
    base = pathlib.Path(parsed.path).name  # e.g. RnRGeekShow_1033.mp3

    # Keep the extension, but sanitize only the stem
    stem, ext = os.path.splitext(base)

    # Calculate how much space we have for the stem (reserve space for extension)
    available_length = max_length - len(ext)

    # Sanitize the stem without length limit first, then truncate
    safe_stem = sanitize_name(stem, max_length=available_length).replace(' ', '_')

    # Build the final filename
    result = f"{safe_stem}{ext}"

    return result


def episode_already_exists(episode_dir: pathlib.Path) -> bool:
    """
    Check if the episode directory exists and contains:
    - metadata.json (valid JSON)
    - description.txt (non-empty)
    - at least one audio file (.mp3, .m4a, .wav, etc.)
    """
    if not episode_dir.exists():
        return False

    # Check for metadata.json
    metadata_file = episode_dir / "metadata.json"
    if not metadata_file.is_file():
        return False
    try:
        with open(metadata_file, encoding="utf-8") as f:
            json.load(f)
    except Exception:
        return False

    # Check for description.txt
    desc_file = episode_dir / "description.txt"
    if not desc_file.is_file():
        return False

    # Check for at least one audio/video file
    audio_extensions = {".mp3", ".m4a", ".mp4", ".wav", ".ogg", ".flac", ".aac"}
    has_audio = any(
        f.suffix.lower() in audio_extensions
        for f in episode_dir.iterdir()
        if f.is_file()
    )

    return has_audio


def download_file(url: str, destination: pathlib.Path) -> None:
    """
    Stream-download *url* and write it to *destination*.
    Raises RuntimeError on network problems.
    """
    headers = {"User-Agent": get_random_user_agent()}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=HTTP_TIMEOUT) as resp:
            resp.raise_for_status()
            with open(destination, "wb") as outf:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        outf.write(chunk)
    except Exception as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc


def process_episode(page_url: str, root_dir: pathlib.Path) -> None:
    """Download page, parse its JSON, and save all artefacts."""
    logger.info(f"Processing: {page_url}")

    # 1. Grab the episode page
    headers = {"User-Agent": get_random_user_agent()}
    try:
        resp = requests.get(page_url, headers=headers, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        logger.warning(f"Failed to fetch {page_url}: {exc}")
        return

    # 2. Find the JSON script
    soup = BeautifulSoup(html, "lxml")
    script_tag = soup.find("script", type="application/ld+json")
    if not script_tag:
        logger.warning(f"No <script type='application/ld+json'> in {page_url}")
        return

    try:
        js_text = (script_tag.string or script_tag.text or "").strip()
        data = json.loads(js_text)
    except Exception as exc:
        logger.warning(f"JSON error in {page_url}: {exc}")
        return

    # 3. Pull fields we care about
    name         = data.get("name")
    description  = (data.get("description") or "").strip()
    content_url  = data.get("associatedMedia", {}).get("contentUrl")
    date_publish = data.get("datePublished")

    # Replace empty description with "No Description"
    if not description:
        description = "No Description"

    if not (name and content_url and date_publish):
        logger.warning(f"Missing required fields in {page_url}")
        return

    # 4. Folder name (date + safe episode name)
    raw_folder_name = f"{date_publish} - {name}"
    folder_name     = sanitize_name(raw_folder_name)
    episode_dir     = root_dir / folder_name

    # Check if episode already exists with all required files
    if episode_already_exists(episode_dir):
        logger.info(f"SKIP - Episode already exists: {episode_dir.name}")
        return
    else:
        logger.info(f"Downloading episode assets to folder: {episode_dir.name}")

    episode_dir.mkdir(parents=True, exist_ok=True)

    # 5. Write metadata.json (full dump)
    metadata_file = episode_dir / "metadata.json"
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning(f"Could not write metadata to {metadata_file}: {exc}")

    # 6. Write description.txt
    desc_file = episode_dir / "description.txt"
    try:
        with open(desc_file, "w", encoding="utf-8") as f:
            f.write(description)
    except Exception as exc:
        logger.warning(f"Could not write description to {desc_file}: {exc}")

    # 7. Download media using the original filename from contentUrl
    mp3_name   = filename_from_contenturl(content_url)
    media_file = episode_dir / mp3_name

    try:
        download_file(content_url, media_file)
        logger.info(f"Media saved to {media_file}")
    except Exception as exc:
        logger.warning(f"Media download failed: {exc}")


def main() -> None:
    """
    Complete workflow:
    1. Download podcast list pages from Podbean
    2. Extract episode URLs from downloaded pages
    3. Download each episode's metadata, description, and audio file
    """
    # Step 1: Download podcast list pages from Podbean
    logger.info("=" * 70)
    logger.info("STEP 1: Downloading podcast list pages from Podbean")
    logger.info("=" * 70)
    download_podcast_pages(BASE_URL, NUM_PAGES, DEFAULT_INPUT_DIR, PAGE_DELAY_IN_SEC)

    # Step 2: Collect episode URLs from HTML pages and generate episode_links.json
    logger.info("=" * 70)
    logger.info("STEP 2: Extracting episode URLs from downloaded pages")
    logger.info("=" * 70)
    collect_episode_urls(DEFAULT_INPUT_DIR, INPUT_JSON)

    # Step 3: Load the list of URLs
    logger.info("=" * 70)
    logger.info("STEP 3: Downloading episodes")
    logger.info("=" * 70)
    try:
        with open(INPUT_JSON, encoding="utf-8") as f:
            urls = json.load(f)
    except Exception as exc:
        logger.error(f"Cannot read {INPUT_JSON}: {exc}")
        return

    if not isinstance(urls, list):
        logger.error(f"{INPUT_JSON} must contain a JSON array of URLs")
        return

    # Randomize the order of URLs to make download pattern less predictable
    random.shuffle(urls)
    logger.info(f"Randomized {len(urls)} episode URLs for processing")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(urls, start=1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Episode {i}/{len(urls)}")
        logger.info(f"{'='*70}")
        process_episode(url, OUTPUT_ROOT)
        time.sleep(DELAY_IN_SEC)

    logger.info("=" * 70)
    logger.info("COMPLETE: All episodes processed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
