#!/usr/bin/env python3
"""Download tiktoken BPE encoding files into the package data directory.

Run this once before building the wheel or running tests:
    python scripts/download_encodings.py

The script checks tiktoken's local cache first to avoid redundant downloads.
Files are placed in src/contextly/tokenizer/data/ and bundled into the wheel
via hatchling's default package-data inclusion.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

_ENCODING_URLS: dict[str, str] = {
    "cl100k_base": ("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"),
    "o200k_base": ("https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"),
}

_DATA_DIR = Path(__file__).parent.parent / "src" / "contextly" / "tokenizer" / "data"


def _tiktoken_cache_path(url: str) -> Path:
    """Return the path where tiktoken caches a given URL (SHA-256 of URL)."""
    cache_dir = Path(os.environ.get("TIKTOKEN_CACHE_DIR", Path.home() / ".cache" / "tiktoken"))
    return cache_dir / hashlib.sha256(url.encode()).hexdigest()


def download_all() -> None:
    """Download all encoding files, using tiktoken cache when available."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in _ENCODING_URLS.items():
        dest = _DATA_DIR / f"{name}.tiktoken"
        if dest.exists():
            print(f"  {name}: already present ({dest.stat().st_size:,} bytes) — skipping")
            continue

        cached = _tiktoken_cache_path(url)
        if cached.exists():
            shutil.copy(cached, dest)
            print(f"  {name}: copied from tiktoken cache ({dest.stat().st_size:,} bytes)")
        else:
            print(f"  {name}: downloading from {url} ...")
            urllib.request.urlretrieve(url, dest)
            print(f"  {name}: done ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    print("Downloading tiktoken encoding files ...")
    download_all()
    print("Done.")
