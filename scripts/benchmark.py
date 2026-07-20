#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepShield Benchmark — measure accuracy, FPR, and FNR.

Usage:
    # Benchmark with default paths (./benchmark_data/{real,fake,ai_generated})
    python scripts/benchmark.py

    # Custom paths
    python scripts/benchmark.py --real-dir ./my_real --fake-dir ./my_fake --ai-dir ./my_ai

    # Export results
    python scripts/benchmark.py --output benchmark_results.json

Directory layout expected:
    benchmark_data/
    ├── real/          # Authentic / real face images
    ├── fake/          # Face-swap deepfakes
    └── ai_generated/  # Fully AI-generated images (ChatGPT, Midjourney, SDXL, etc.)

Each directory should contain .jpg, .png, .jpeg files.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

# ─── Disable verbose logging for benchmark ──────────────────
import loguru as _loguru
_loguru.logger.remove()
_loguru.logger.add(lambda _: None, level="WARNING")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepShield Benchmark")
    parser.add_argument(
        "--real-dir",
        type=str,
        default="benchmark_data/real",
        help="Directory with authentic (real) face images",
    )
    parser.add_argument(
        "--fake-dir",
        type=str,
        default="benchmark_data/fake",
        help="Directory with face-swap deepfake images",
    )
    parser.add_argument(
        "--ai-dir",
        type=str,
        default="benchmark_data/ai_generated",
        help="Directory with AI-generated (synthetic) images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write benchmark results as JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device (auto / cpu / cuda)",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=0,
        help="Max images per category (0 = all)",
    )
    return parser.parse_args()


def load_images(directory: str, max_count: int = 0) -> list[tuple[str, str]]:
    """Load image paths from a directory."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    path = Path(directory)
    if not path.is_dir():
        print(f"  ⚠  Directory not found: {directory}")
        return []

    files = sorted(
        [str(f) for f in path.iterdir() if f.suffix.lower() in exts]
    )
    if max_count > 0:
        files = files[:max_count]
    return files


def benchmark_category(
    pipeline,
    file_paths: list[str],
    category: str,
) -> dict[str, Any]:
    """Run all images in a category through the pipeline."""
    from src.core.pipelines.image_pipeline import ImageDetectionPipeline

    results = []
    correct = 0
    total = len(file_paths)

    for i, path in enumerate(file_paths):
        try:
            result = pipeline.detect(path)
            pred = result.prediction  # "REAL" or "FAKE"
            d = result.to_dict()

            # For "real" category, correct = REAL prediction (true negative for deepfake)
            # For "fake"/"ai_generated" categories, correct = FAKE prediction (true positive)
            if category == "real":
                is_correct = pred == "REAL"
            else:
                is_correct = pred == "FAKE"

            if is_correct:
                correct += 1

            results.append({
                "file": path,
                "prediction": pred,
                "confidence": d["confidence"],
                "probability": d["probability"],
                "ai_gen_score": d.get("ai_gen_score"),
                "face_forgery_score": d.get("face_forgery_score"),
                "correct": is_correct,
            })

            status = "✓" if is_correct else "✗"
            print(f"  [{i+1}/{total}] {status} {Path(path).name} → {pred} (conf={d['confidence']:.2f})")

        except Exception as e:
            print(f"  [{i+1}/{total}] ⚠ {Path(path).name} → ERROR: {e}")
            results.append({"file": path, "error": str(e), "correct": False})

    accuracy = correct / total if total > 0 else 0.0
    return {
        "category": category,
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": round(accuracy, 4),
    }, results


def print_summary(all_stats: list[dict[str, Any]]) -> None:
    """Print a formatted benchmark summary."""
    print()
    print("=" * 70)
    print("  🛡️  DeepShield — Benchmark Results")
    print("=" * 70)
    print()

    # Per-category breakdown
    print(f"{'Category':<20} {'Total':>8} {'Correct':>8} {'Wrong':>8} {'Accuracy':>10}")
    print("-" * 60)

    total_all = 0
    correct_all = 0
    for stat in all_stats:
        cat = stat["category"]
        if stat["total"] == 0:
            print(f"{cat:<20} {'—':>8} {'—':>8} {'—':>8} {'—':>10}")
            continue
        acc = stat["accuracy"]
        print(
            f"{cat:<20} {stat['total']:>8} {stat['correct']:>8} "
            f"{stat['incorrect']:>8} {acc:>8.1%}"
        )
        total_all += stat["total"]
        correct_all += stat["correct"]

    print("-" * 60)

    # Compute FPR / FNR
    real_stat = next((s for s in all_stats if s["category"] == "real"), None)
    fake_stat = next((s for s in all_stats if s["category"] == "fake"), None)
    ai_stat = next((s for s in all_stats if s["category"] == "ai_generated"), None)

    overall_acc = correct_all / total_all if total_all > 0 else 0.0
    print(f"{'OVERALL':<20} {total_all:>8} {correct_all:>8} {total_all - correct_all:>8} {overall_acc:>8.1%}")

    print()
    print("-" * 70)
    if real_stat and real_stat["total"] > 0:
        fpr = real_stat["incorrect"] / real_stat["total"]
        print(f"  False Positive Rate (FPR):  {fpr:.2%}  ({real_stat['incorrect']}/{real_stat['total']} real images flagged as FAKE)")
    else:
        print("  False Positive Rate (FPR):  N/A (no real images tested)")

    if fake_stat and fake_stat["total"] > 0:
        fnr = fake_stat["incorrect"] / fake_stat["total"]
        print(f"  False Negative Rate (FNR):  {fnr:.2%}  ({fake_stat['incorrect']}/{fake_stat['total']} fake images missed)")
    else:
        print("  False Negative Rate (FNR):  N/A (no fake images tested)")

    if ai_stat and ai_stat["total"] > 0:
        ai_fnr = ai_stat["incorrect"] / ai_stat["total"]
        print(f"  AI-Gen Miss Rate:           {ai_fnr:.2%}  ({ai_stat['incorrect']}/{ai_stat['total']} AI images missed)")
    else:
        print("  AI-Gen Miss Rate:           N/A (no AI-generated images tested)")

    print("=" * 70)
    print()


def main() -> None:
    args = parse_args()

    # ── Load images ──
    print("\n📂 Loading benchmark images...")
    real_images = load_images(args.real_dir, args.max_per_category)
    fake_images = load_images(args.fake_dir, args.max_per_category)
    ai_images = load_images(args.ai_dir, args.max_per_category)

    print(f"   Real:           {len(real_images)} images")
    print(f"   Face-swap fake: {len(fake_images)} images")
    print(f"   AI-generated:   {len(ai_images)} images")

    if not any([real_images, fake_images, ai_images]):
        print("\n⚠  No images found. Create benchmark_data/ directories or use --real-dir etc.")
        print("   Expected layout:")
        print("     benchmark_data/real/          — authentic face images")
        print("     benchmark_data/fake/          — face-swap deepfakes")
        print("     benchmark_data/ai_generated/  — AI-generated stills")
        return

    # ── Initialize pipeline ──
    print("\n🔧 Initializing DeepShield pipeline...")
    from src.core.pipelines.image_pipeline import ImageDetectionPipeline

    pipeline = ImageDetectionPipeline(device=args.device)
    pipeline.initialize()

    # ── Run benchmarks ──
    all_stats = []
    all_results = []

    if real_images:
        print(f"\n📷 Benchmarking: REAL images ({len(real_images)} files)")
        stat, results = benchmark_category(pipeline, real_images, "real")
        all_stats.append(stat)
        all_results.extend(results)

    if fake_images:
        print(f"\n🎭 Benchmarking: FACE-SWAP fake images ({len(fake_images)} files)")
        stat, results = benchmark_category(pipeline, fake_images, "fake")
        all_stats.append(stat)
        all_results.extend(results)

    if ai_images:
        print(f"\n🤖 Benchmarking: AI-GENERATED images ({len(ai_images)} files)")
        stat, results = benchmark_category(pipeline, ai_images, "ai_generated")
        all_stats.append(stat)
        all_results.extend(results)

    # ── Print summary ──
    print_summary(all_stats)

    # ── Export ──
    if args.output:
        output_data = {
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": args.device,
            "stats": all_stats,
            "results": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"📄 Results exported to {args.output}")


if __name__ == "__main__":
    main()
