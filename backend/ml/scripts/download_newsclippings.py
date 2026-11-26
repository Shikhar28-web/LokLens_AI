"""
Robust chunked downloader for NewsCLIPpings dataset with progress reporting.
"""
import os
import sys
import time
import logging
from pathlib import Path
import urllib.request

sys.path.insert(0, '.')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://huggingface.co/g-luo/news-clippings/resolve/main"
LOCAL_DIR = Path("data/datasets/newsclippings/news_clippings")

def download_file_chunked(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("content-length", 0))
        total_mb = total_size / (1024 * 1024)

        if dest_path.exists() and total_size > 0 and dest_path.stat().st_size == total_size:
            logger.info("Already downloaded & verified: %s (%.1f MB)", dest_path.name, total_mb)
            return

        logger.info("Downloading: %s → %s (%.1f MB)", url, dest_path.name, total_mb)
        downloaded = 0
        last_logged = time.time()
        chunk_size = 1024 * 1024  # 1 MB chunks

        with open(temp_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if time.time() - last_logged > 5:
                    done_mb = downloaded / (1024 * 1024)
                    pct = (downloaded / total_size * 100) if total_size else 0
                    logger.info("  Progress: %.1f / %.1f MB (%.1f%%)", done_mb, total_mb, pct)
                    last_logged = time.time()

    if temp_path.exists():
        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)
    logger.info("Saved: %s (%.1f MB)", dest_path.name, dest_path.stat().st_size / (1024 * 1024))

def download_newsclippings():
    logger.info("=== Downloading NewsCLIPpings dataset ===")

    # 1. Annotations
    for split in ["train", "val", "test"]:
        rel = f"data/merged_balanced/{split}.json"
        dest = LOCAL_DIR / rel
        download_file_chunked(f"{BASE_URL}/{rel}", dest)

    # 2. Precomputed CLIP embeddings
    # Prioritize val & test (quick, 100MB each) then train (1GB)
    for split in ["val", "test", "train"]:
        for emb_type in ["clip_image_embeddings", "clip_text_embeddings"]:
            rel = f"embeddings/{emb_type}/{emb_type}_{split}.pkl"
            dest = LOCAL_DIR / rel
            download_file_chunked(f"{BASE_URL}/{rel}", dest)

    logger.info("=== All NewsCLIPpings files downloaded successfully! ===")

if __name__ == "__main__":
    download_newsclippings()
