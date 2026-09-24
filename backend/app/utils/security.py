"""
Security utilities — file validation, SSRF protection, filename sanitization.
Never trust user input.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
import socket
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from fastapi import HTTPException, UploadFile, status

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

# ── File upload validation 

ALLOWED_MAGIC_BYTES: dict[str, bytes] = {
    "image/jpeg": b"\xff\xd8\xff",
    "image/png": b"\x89PNG",
    "image/webp": b"RIFF",
    "image/gif": b"GIF8",
}

_FILENAME_STRIP = re.compile(r"[^\w.\-]")


def sanitize_filename(filename: str | None) -> str:
    """Strip all characters except alphanumeric, dots, hyphens."""
    if not filename:
        return "upload"
    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower()
    clean = _FILENAME_STRIP.sub("_", stem)[:64]
    return f"{clean}{suffix}" if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"} else clean


async def validate_image_upload(upload: UploadFile, settings: "Settings") -> bytes:
    """
    Read, validate MIME type (by magic bytes), and enforce size limit.
    Returns raw bytes if valid; raises HTTPException otherwise.
    """
    # Reject declared type immediately if not in allowlist
    declared = (upload.content_type or "").lower().split(";")[0].strip()
    if declared not in settings.allowed_image_types_list:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {declared}. Allowed: {settings.allowed_image_types_list}",
        )

    content = await upload.read()

    # Enforce size limit
    if len(content) > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {settings.max_image_size_bytes // 1_048_576} MB limit.",
        )

    # Verify magic bytes match declared type
    for mime, magic in ALLOWED_MAGIC_BYTES.items():
        if declared == mime and not content.startswith(magic):
            # WEBP has RIFF....WEBP structure
            if mime == "image/webp" and content[8:12] == b"WEBP":
                continue
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="File content does not match declared MIME type.",
            )

    return content


# ── SSRF protection ────────────────────────────────────────────────────────────

_PRIVATE_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # link-local
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def is_safe_url(url: str, allowed_schemes: list[str]) -> bool:
    """
    Returns True if the URL is safe to fetch:
    - Uses an allowed scheme (http/https)
    - Does not resolve to a private/loopback IP (SSRF protection)
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in allowed_schemes:
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        # Resolve hostname to IP
        try:
            addr = socket.gethostbyname(hostname)
        except socket.gaierror:
            return False
        ip = ipaddress.ip_address(addr)
        for private in _PRIVATE_RANGES:
            if ip in private:
                logger.warning("SSRF blocked: %s resolves to private IP %s", url, ip)
                return False
        return True
    except Exception as exc:
        logger.warning("URL safety check failed for %s: %s", url, exc)
        return False


# ── IP hashing (for rate limiting without storing raw IPs) ────────────────────

def hash_ip(ip: str) -> str:
    """One-way hash of an IP address for rate-limit keying."""
    return hashlib.sha256(ip.encode()).hexdigest()
