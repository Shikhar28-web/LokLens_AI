import hashlib
import json
import logging
from typing import Dict, Any, Tuple
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS
import imagehash
import piexif

logger = logging.getLogger(__name__)


def extract_image_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extract metadata and cryptographic/perceptual hashes from an image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Cryptographic Hash (SHA-256)
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    sha256_hash = sha256.hexdigest()

    # Perceptual Hashes and Dimensions
    try:
        with Image.open(path) as img:
            width, height = img.size
            # Convert to RGB to ensure hashing consistency
            if img.mode != "RGB":
                rgb_img = img.convert("RGB")
            else:
                rgb_img = img

            p_hash = str(imagehash.phash(rgb_img))
            d_hash = str(imagehash.dhash(rgb_img))
            a_hash = str(imagehash.average_hash(rgb_img))

            # EXIF extraction
            exif_data = _extract_exif(img)
            
    except Exception as e:
        logger.error(f"Failed to process image {image_path}: {e}")
        width, height = 0, 0
        p_hash, d_hash, a_hash = "", "", ""
        exif_data = {}

    return {
        "file_size_bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "sha256_hash": sha256_hash,
        "phash": p_hash,
        "dhash": d_hash,
        "ahash": a_hash,
        "exif_json": exif_data
    }


def _extract_exif(img: Image.Image) -> Dict[str, Any]:
    """Helper to safely extract and stringify EXIF data."""
    exif_dict = {}
    try:
        if "exif" in img.info:
            exif_raw = piexif.load(img.info["exif"])
            
            # Combine 0th, Exif, GPS, 1st
            for ifd in ("0th", "Exif", "GPS", "1st"):
                for tag_id, value in exif_raw.get(ifd, {}).items():
                    tag_name = piexif.TAGS[ifd].get(tag_id, {"name": str(tag_id)})["name"]
                    
                    # Convert bytes to string safely
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8").strip('\x00')
                        except UnicodeDecodeError:
                            value = value.hex()
                            
                    # Convert tuples (often rationals) to string representations
                    if isinstance(value, tuple):
                        if len(value) == 2 and isinstance(value[0], int) and isinstance(value[1], int):
                            if value[1] == 0:
                                value = "0"
                            else:
                                value = f"{value[0]}/{value[1]}"
                        else:
                            value = str(value)
                            
                    exif_dict[tag_name] = value
                    
    except Exception as e:
        logger.warning(f"Error parsing EXIF: {e}")
        
    return exif_dict
