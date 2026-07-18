#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Pre-trained Models for DeepShield.

Usage:
    python scripts/download_models.py           # Download all
    python scripts/download_models.py --light    # Download light version
    python scripts/download_models.py --list     # List available models

This script downloads pre-trained weights from HuggingFace Hub
and other sources. Models are placed in models/weights/.
"""

from __future__ import annotations

import hashlib
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

MODELS_DIR = Path("models/weights")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Model Registry ─────────────────────────────────────
MODELS = {
    "xception_ffpp": {
        "url": "https://huggingface.co/RamadhanZome/deepfake-xception/resolve/main/best_xception.pth",
        "filename": "xception_ffpp.pt",
        "description": "XceptionNet trained on FaceForensics++ (99.36% acc)",
        "size_mb": 98,
        "sha256": None,
        "required": True,
    },
    "efficientnet_celebdf": {
        "url": "https://huggingface.co/viktorahnstrom/xade-deepfake-detector/resolve/main/best_model.pt",
        "filename": "efficientnet_celebdf.pt",
        "description": "EfficientNet-B4 trained on Celeb-DF + FF++ + CIPLAB",
        "size_mb": 183,
        "sha256": None,
        "required": True,
    },
    "vit_face_forensics": {
        "url": "https://huggingface.co/google/vit-base-patch16-224/resolve/main/pytorch_model.bin",
        "filename": "vit_face_forensics.pt",
        "description": "Vision Transformer (ViT-B/16) base weights",
        "size_mb": 330,
        "sha256": None,
        "required": False,
    },
    "whisper_base": {
        "url": "https://huggingface.co/openai/whisper-base/resolve/main/pytorch_model.bin",
        "filename": "whisper_base.pt",
        "description": "OpenAI Whisper base model (74M params) for audio analysis",
        "size_mb": 277,
        "sha256": None,
        "required": True,
    },
}

LIGHT_MODELS = {
    "xception_ffpp": MODELS["xception_ffpp"],
    "whisper_base": MODELS["whisper_base"],
}


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=desc or dest.name,
            ncols=80,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    return dest


def extract_bz2(file_path: Path) -> Path:
    """Extract .bz2 file."""
    import bz2
    import shutil

    output_path = file_path.with_suffix("")  # Remove .bz2
    with bz2.BZ2File(file_path, "rb") as src, open(output_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    file_path.unlink()  # Remove compressed file
    return output_path


def verify_checksum(file_path: Path, expected: Optional[str]) -> bool:
    """Verify SHA-256 checksum of a file."""
    if expected is None:
        return True

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest() == expected


def main(light: bool = False, list_only: bool = False) -> None:
    """Main download routine."""
    if list_only:
        print("\n📦 Available Pre-trained Models:\n")
        for name, info in MODELS.items():
            req = "[REQUIRED]" if info["required"] else "[OPTIONAL]"
            print(f"  {name:30s} {req} {info['size_mb']:4d}MB - {info['description']}")
        print()
        return

    models = LIGHT_MODELS if light else MODELS
    total_models = len(models)
    total_size_mb = sum(m["size_mb"] for m in models.values())

    print(f"\n{'='*60}")
    print(f"  🛡️  DeepShield — Model Downloader")
    print(f"  {'Light mode' if light else 'Full download'}")
    print(f"  {total_models} models, ~{total_size_mb}MB total")
    print(f"{'='*60}\n")

    downloaded = 0
    failed = 0

    for name, info in models.items():
        dest = MODELS_DIR / info["filename"]

        if dest.exists():
            if verify_checksum(dest, info.get("sha256")):
                print(f"  ✓ {name}: already exists ({dest})")
                downloaded += 1
                continue
            else:
                print(f"  ⚠ {name}: checksum mismatch, re-downloading...")

        try:
            print(f"  ↓ {name}: {info['description']}")
            download_file(info["url"], dest, desc=f"    {name}")

            # Handle compressed files
            if info.get("compressed"):
                print(f"    Extracting...")
                dest = extract_bz2(dest)
                print(f"    Extracted to: {dest}")

            # Verify
            if verify_checksum(dest, info.get("sha256")):
                print(f"  ✓ {name}: {info['size_mb']}MB -> {dest}")
                downloaded += 1
            else:
                print(f"  ✗ {name}: checksum verification failed")
                failed += 1

        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1

        print()

    # Summary
    print(f"{'='*60}")
    print(f"  Results: {downloaded} downloaded, {failed} failed")
    print(f"  Model directory: {MODELS_DIR.resolve()}")
    print(f"{'='*60}\n")

    if failed == 0:
        print("  ✅ All models ready!\n")
    else:
        print(f"  ⚠️  {failed} model(s) failed. Check logs.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download DeepShield pre-trained models")
    parser.add_argument("--light", action="store_true", help="Download only essential models")
    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()
    main(light=args.light, list_only=args.list)
