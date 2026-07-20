#!/usr/bin/env python3
"""
Industry-standard domain adaptation for modern face-swaps (InsightFace / inswapper).

Transfer-learns from Celeb-DF EfficientNet with a replay buffer of diverse real faces
to avoid catastrophic forgetting (standard continual-learning practice).

Usage:
    python scripts/finetune_faceswap.py
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import efficientnet_b4


class FaceSwapDataset(Dataset):
    def __init__(self, root: Path, split: str = "train", val_ratio: float = 0.2) -> None:
        samples: list[tuple[Path, int, str]] = []
        for label, name in [(0, "real"), (1, "fake")]:
            folder = root / name
            for p in sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png"))):
                # tag: web replay vs dataset
                tag = "web" if p.name.startswith(("real_", "aug_", "web_")) else "ds"
                samples.append((p, label, tag))

        rng = random.Random(42)
        rng.shuffle(samples)
        cut = int(len(samples) * (1 - val_ratio))
        self.samples = samples[:cut] if split == "train" else samples[cut:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, tag = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read {path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (380, 380), interpolation=cv2.INTER_LANCZOS4)

        if random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        if random.random() < 0.5:
            rgb = np.clip(rgb.astype(np.float32) * random.uniform(0.85, 1.15), 0, 255).astype(
                np.uint8
            )
        if random.random() < 0.3:
            noise = np.random.randn(*rgb.shape).astype(np.float32) * 3
            rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        # sample weight: upweight web reals to protect FPR
        w = 2.5 if (label == 0 and tag == "web") else 1.0
        return tensor, torch.tensor(label, dtype=torch.long), torch.tensor(w, dtype=torch.float32)


def build_model(device: str, base_ckpt: Path) -> nn.Module:
    model = efficientnet_b4(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1792, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(512, 2),
    )
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cleaned = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)

    for _, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(k in name for k in ("features.6", "features.7", "features.8", "classifier")):
            param.requires_grad = True
    return model.to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        prob = F.softmax(model(x), dim=-1)[:, 1].cpu().numpy()
        y_prob.extend(prob.tolist())
        y_true.extend(y.cpu().numpy().tolist())

    y_true_a = np.array(y_true)
    y_prob_a = np.array(y_prob)
    preds = (y_prob_a >= 0.5).astype(int)
    acc = float((preds == y_true_a).mean()) if len(y_true_a) else 0.0

    # Sweep threshold for best F1 on this val split
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.3, 0.8, 26):
        p = (y_prob_a >= t).astype(int)
        tp = int(((p == 1) & (y_true_a == 1)).sum())
        fp = int(((p == 1) & (y_true_a == 0)).sum())
        fn = int(((p == 0) & (y_true_a == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    fake_mean = float(y_prob_a[y_true_a == 1].mean()) if (y_true_a == 1).any() else 0.0
    real_mean = float(y_prob_a[y_true_a == 0].mean()) if (y_true_a == 0).any() else 0.0
    return {
        "acc": acc,
        "f1": float(best_f1),
        "threshold": best_t,
        "fake_mean": fake_mean,
        "real_mean": real_mean,
        "sep": fake_mean - real_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="benchmark_data/finetune")
    parser.add_argument("--base-ckpt", default="models/weights/efficientnet_celebdf.pt")
    parser.add_argument("--output", default="models/weights/efficientnet_inswapper.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    data_dir = Path(args.data_dir)

    train_ds = FaceSwapDataset(data_dir, "train")
    val_ds = FaceSwapDataset(data_dir, "val")
    print(f"Train={len(train_ds)} Val={len(val_ds)} device={device}")

    # Weighted sampler: boost web-real replay samples
    sample_weights = []
    for _, label, tag in train_ds.samples:
        if label == 0 and tag == "web":
            sample_weights.append(3.0)
        elif label == 1:
            sample_weights.append(2.0)
        else:
            sample_weights.append(1.0)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(device, Path(args.base_ckpt))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_score = -1.0
    best_state = None
    best_threshold = 0.5
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y, w in train_loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            opt.zero_grad()
            logits = model(x)
            per = F.cross_entropy(logits, y, reduction="none", label_smoothing=0.05)
            loss = (per * w).mean()
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(y)
            n += len(y)
        sched.step()

        metrics = evaluate(model, val_loader, device)
        # Optimize separation + F1 (industry calibration objective)
        score = metrics["f1"] + 0.25 * metrics["sep"]
        print(
            f"Epoch {epoch}/{args.epochs} loss={running/max(n,1):.4f} "
            f"acc={metrics['acc']:.3f} f1={metrics['f1']:.3f} "
            f"t*={metrics['threshold']:.2f} sep={metrics['sep']:.3f} "
            f"fakeμ={metrics['fake_mean']:.2f} realμ={metrics['real_mean']:.2f}"
        )
        if score >= best_score:
            best_score = score
            best_threshold = metrics["threshold"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "class_names": ["real", "fake"],
            "epoch": args.epochs,
            "val_score": round(best_score, 4),
            "decision_threshold": round(best_threshold, 3),
            "base_checkpoint": str(args.base_ckpt),
            "adaptation": "inswapper_insightface_with_real_replay",
            "train_seconds": round(time.time() - start, 1),
        },
        out,
    )
    print(f"Saved → {out}  best_score={best_score:.3f} threshold={best_threshold:.3f}")


if __name__ == "__main__":
    main()
