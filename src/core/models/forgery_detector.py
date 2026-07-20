"""
Forgery Detection using pre-trained vision models.

Primary detection layer using fine-tuned XceptionNet/EfficientNet
for face forgery classification.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.core.models.base import BaseModel, ModelOutput, ModelType, ModelRegistry


class XceptionBlock(nn.Module):
    """Xception-style residual block for feature extraction."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ForgeryDetectorNet(nn.Module):
    """
    Lightweight Xception-inspired network for deepfake detection.

    Architecture:
    - Depthwise separable convolutions (efficient)
    - Residual connections (stable training)
    - Global average pooling (resolution agnostic)
    - Binary classification head
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()

        # Entry flow
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Middle flow (4 blocks)
        self.middle = nn.Sequential(
            XceptionBlock(32, 64),
            XceptionBlock(64, 128, stride=2),
            XceptionBlock(128, 256, stride=2),
            XceptionBlock(256, 512, stride=2),
        )

        # Exit flow
        self.exit = nn.Sequential(
            XceptionBlock(512, 1024, stride=2),
            nn.Conv2d(1024, 1024, 3, padding=1, groups=1024, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Feature extractor (for ensemble)
        self.features = nn.Sequential(self.entry, self.middle, self.exit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        feat = self.features(x)
        return self.classifier(feat)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings (before classification head)."""
        feat = self.features(x)
        return F.adaptive_avg_pool2d(feat, 1).flatten(1)


@ModelRegistry.register
class ForgeryDetector(BaseModel):
    """
    Face-forgery detection using production-style multi-backbone ensemble.

    Default ``backbone="ensemble"`` mirrors high-accuracy open-source systems:
    EfficientNet-B4 (Celeb-DF) 55% + Xception (trained detector) 45%.
    """

    model_type = ModelType.FORGERY_DETECTION
    model_name = "forgery_detector"

    DEFAULT_WEIGHT_PATHS = {
        "efficientnet": "models/weights/efficientnet_celebdf.pt",
        "efficientnet_inswapper": "models/weights/efficientnet_inswapper.pt",
        "xception": "models/weights/xception_ffpp.pt",
        "vit": "models/weights/vit_face_forensics.pt",
    }
    # Industry soft-vote mix: adapted InsightFace specialist dominates when present
    ENSEMBLE_WEIGHTS = {
        "efficientnet_inswapper": 0.55,
        "efficientnet": 0.30,
        "xception": 0.15,
    }

    def __init__(
        self,
        backbone: str = "ensemble",
        device: str = "auto",
        use_pretrained: bool = True,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__(device=device)
        self.backbone = backbone
        self.use_pretrained = use_pretrained
        self._weights_path = weights_path
        self._net = None
        self._nets: dict[str, nn.Module] = {}
        self._net_meta: dict[str, dict[str, Any]] = {}
        self._transform = None
        self._input_size = 256
        # Index mapping used by predict(); overwritten when a checkpoint declares class names
        self._class_to_idx = {"real": 0, "fake": 1}
        self.load_weights(weights_path)

    def load_weights(self, weights_path: Optional[str] = None) -> None:
        """Load forgery detection model and trained checkpoint when available."""
        from pathlib import Path
        import time

        start = time.perf_counter()
        weights_path = weights_path or self._weights_path

        if self.backbone == "ensemble":
            self._load_ensemble_backbones()
            self._input_size = 380
            self._transform_params = {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }
            self._is_loaded = True
            self._load_time_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"Forgery detector (ensemble={list(self._nets.keys())}) "
                f"loaded in {self._load_time_ms:.1f}ms"
            )
            return

        if weights_path is None:
            default = Path(self.DEFAULT_WEIGHT_PATHS.get(self.backbone, ""))
            if default.exists():
                weights_path = str(default)

        if self.backbone == "efficientnet":
            self._net = self._build_efficientnet()
            self._input_size = 380
            self._transform_params = {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }
            if weights_path:
                self._load_efficientnet_checkpoint(weights_path)
            elif self.use_pretrained:
                logger.warning(
                    "EfficientNet deepfake weights missing; using ImageNet features only"
                )

        elif self.backbone == "xception":
            from src.core.models.xception_net import XceptionNet

            self._net = XceptionNet(num_classes=2)
            self._input_size = 299
            # RamadhanZome / Chollet Xception training norm + class order
            self._transform_params = {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            }
            self._class_to_idx = {"fake": 0, "real": 1}
            if weights_path:
                loaded = self._try_load_state_dict(weights_path, strict=True)
                if loaded:
                    logger.info(f"Loaded Xception weights from {weights_path}")
                else:
                    logger.warning(
                        "Xception checkpoint architecture mismatch; using random weights"
                    )
            else:
                logger.warning("No Xception checkpoint found; using random weights")

        elif self.backbone == "vit":
            from transformers import ViTForImageClassification, ViTConfig

            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_hidden_layers=12,
                hidden_size=768,
                num_attention_heads=12,
                num_labels=2,
            )
            self._net = ViTForImageClassification(config)
            self._input_size = 224
            self._transform_params = {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }
            if weights_path:
                self._try_load_state_dict(weights_path, strict=False)
            elif self.use_pretrained:
                from transformers import ViTModel

                pretrained = ViTModel.from_pretrained("google/vit-base-patch16-224")
                self._net.vit.load_state_dict(pretrained.state_dict(), strict=False)

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        self._net = self._net.to(self.device).eval()
        self._is_loaded = True
        self._load_time_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Forgery detector ({self.backbone}) loaded in {self._load_time_ms:.1f}ms "
            f"[classes={self._class_to_idx}]"
        )

    def _load_ensemble_backbones(self) -> None:
        """Load industry soft-vote backbones (adapted + Celeb-DF + Xception)."""
        from pathlib import Path
        from src.core.models.xception_net import XceptionNet

        self._decision_threshold = 0.5

        # 1) Domain-adapted InsightFace / inswapper specialist (preferred)
        ins_path = Path(self.DEFAULT_WEIGHT_PATHS["efficientnet_inswapper"])
        if ins_path.exists():
            net = self._build_efficientnet().to(self.device).eval()
            self._net = net
            self._load_efficientnet_checkpoint(str(ins_path))
            thr = 0.5
            try:
                ckpt = torch.load(str(ins_path), map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "decision_threshold" in ckpt:
                    thr = float(ckpt["decision_threshold"])
                    self._decision_threshold = thr
            except Exception:
                pass
            self._nets["efficientnet_inswapper"] = net
            self._net_meta["efficientnet_inswapper"] = {
                "input_size": 380,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "class_to_idx": dict(self._class_to_idx),
                "weight": self.ENSEMBLE_WEIGHTS["efficientnet_inswapper"],
                "decision_threshold": thr,
            }
            logger.info(
                f"Loaded InsightFace-adapted EfficientNet from {ins_path} "
                f"(threshold={thr:.2f})"
            )

        # 2) Original Celeb-DF EfficientNet (general face-swap / FF++)
        eff_path = Path(self.DEFAULT_WEIGHT_PATHS["efficientnet"])
        if eff_path.exists():
            net = self._build_efficientnet().to(self.device).eval()
            self._net = net
            self._load_efficientnet_checkpoint(str(eff_path))
            self._nets["efficientnet"] = net
            self._net_meta["efficientnet"] = {
                "input_size": 380,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "class_to_idx": dict(self._class_to_idx),
                "weight": self.ENSEMBLE_WEIGHTS["efficientnet"],
            }
        else:
            logger.warning("EfficientNet Celeb-DF weights missing from face-forgery ensemble")

        # 3) Xception (StyleGAN / generic detector) — advisory via soft-OR
        xcp_path = Path(self.DEFAULT_WEIGHT_PATHS["xception"])
        if xcp_path.exists():
            net = XceptionNet(num_classes=2).to(self.device).eval()
            self._net = net
            if self._try_load_state_dict(str(xcp_path), strict=True):
                self._nets["xception"] = net
                self._net_meta["xception"] = {
                    "input_size": 299,
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                    "class_to_idx": {"fake": 0, "real": 1},
                    "weight": self.ENSEMBLE_WEIGHTS["xception"],
                }
                logger.info(f"Loaded Xception into forgery ensemble from {xcp_path}")
            else:
                logger.warning("Xception failed to load into forgery ensemble")

        if not self._nets:
            raise RuntimeError("No forgery backbones available for ensemble")
        self._net = next(iter(self._nets.values()))
        self._class_to_idx = self._net_meta[next(iter(self._nets))]["class_to_idx"]

    def _build_efficientnet(self) -> nn.Module:
        from torchvision.models import efficientnet_b4

        base = efficientnet_b4(weights=None)
        # Match XADE / Celeb-DF checkpoint classifier layout
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1792, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(512, 2),
        )
        return base

    def _load_efficientnet_checkpoint(self, weights_path: str) -> None:
        """Load EfficientNet-B4 deepfake detector checkpoint."""
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            class_names = checkpoint.get("class_names")
            if class_names and len(class_names) == 2:
                self._class_to_idx = {
                    str(name).lower(): idx for idx, name in enumerate(class_names)
                }
            logger.info(
                f"EfficientNet checkpoint epoch={checkpoint.get('epoch')} "
                f"val_acc={checkpoint.get('val_acc')}"
            )
        else:
            state_dict = checkpoint

        # Strip optional "model." prefix from DataParallel / wrapper saves
        cleaned = {
            (k[len("model.") :] if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }
        missing, unexpected = self._net.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            logger.warning(
                f"EfficientNet load warnings missing={missing} unexpected={unexpected}"
            )
        else:
            logger.info(f"Loaded EfficientNet deepfake weights from {weights_path}")

    def _try_load_state_dict(self, weights_path: str, strict: bool = False) -> bool:
        try:
            state = torch.load(weights_path, map_location=self.device, weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            missing, unexpected = self._net.load_state_dict(state, strict=strict)
            ok = len(missing) == 0 and len(unexpected) == 0
            if not ok:
                logger.debug(f"Partial load missing={len(missing)} unexpected={len(unexpected)}")
            return ok or (len(missing) < len(state) // 2)
        except Exception as e:
            logger.warning(f"Failed to load weights from {weights_path}: {e}")
            return False

    def _preprocess(
        self,
        image: np.ndarray,
        input_size: Optional[int] = None,
        mean: Optional[list[float]] = None,
        std: Optional[list[float]] = None,
    ) -> torch.Tensor:
        """Convert BGR image to normalized tensor."""
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size = input_size or self._input_size
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LANCZOS4)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        mean_v = mean or self._transform_params["mean"]
        std_v = std or self._transform_params["std"]
        mean_t = torch.tensor(mean_v).view(3, 1, 1)
        std_t = torch.tensor(std_v).view(3, 1, 1)
        tensor = (tensor - mean_t) / std_t
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, input_data: np.ndarray, **kwargs) -> ModelOutput:
        """
        Detect if a face image is real or fake.

        Args:
            input_data: BGR face image (H, W, 3)

        Returns:
            ModelOutput with prediction and confidence
        """
        import time

        start = time.perf_counter()

        if self.backbone == "ensemble" and self._nets:
            member_scores: dict[str, float] = {}
            for name, net in self._nets.items():
                meta = self._net_meta[name]
                tensor = self._preprocess(
                    input_data,
                    input_size=meta["input_size"],
                    mean=meta["mean"],
                    std=meta["std"],
                )
                logits = net(tensor)
                probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
                class_map = meta["class_to_idx"]
                fake_idx = int(class_map.get("fake", 1))
                fake_prob_i = float(probs[fake_idx])
                member_scores[name] = fake_prob_i

            # Industry soft-vote (abraraltaf92 / DeepDect): weighted average of
            # complementary face-forgery specialists, then soft-OR with the peak.
            weight_sum = 0.0
            weighted = 0.0
            for name, score in member_scores.items():
                w = float(self._net_meta[name].get("weight", 0.0))
                if w <= 0:
                    continue
                weighted += w * score
                weight_sum += w
            soft_vote = weighted / weight_sum if weight_sum > 0 else float(np.mean(list(member_scores.values())))
            peak = float(max(member_scores.values()))
            # Prefer adapted specialist peak when present and confident
            if "efficientnet_inswapper" in member_scores:
                adapted = member_scores["efficientnet_inswapper"]
                thr = float(
                    self._net_meta["efficientnet_inswapper"].get("decision_threshold", 0.3)
                )
                if adapted >= thr:
                    fake_prob = float(max(soft_vote, adapted))
                else:
                    fake_prob = float(0.7 * soft_vote + 0.3 * adapted)
            else:
                fake_prob = float(max(soft_vote, 0.85 * peak + 0.15 * soft_vote))

            real_prob = 1.0 - fake_prob
            prediction = "fake" if fake_prob >= 0.5 else "real"
            confidence = max(fake_prob, real_prob)
            elapsed = (time.perf_counter() - start) * 1000
            return ModelOutput(
                prediction=prediction,
                confidence=confidence,
                probabilities={"real": real_prob, "fake": fake_prob},
                metadata={
                    "backbone": "ensemble",
                    "member_fake_probs": member_scores,
                    "fusion": "soft_vote_adapted",
                    "decision_threshold": getattr(self, "_decision_threshold", 0.5),
                    "ensemble_weights": {
                        k: self._net_meta[k]["weight"] for k in self._nets
                    },
                },
                inference_time_ms=elapsed,
            )

        tensor = self._preprocess(input_data)
        result, elapsed = self._measure_inference(self._forward, tensor)

        if hasattr(result, "logits"):
            logits = result.logits
        else:
            logits = result

        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        fake_idx = int(self._class_to_idx.get("fake", 1))
        real_idx = int(self._class_to_idx.get("real", 0))
        fake_prob = float(probs[fake_idx])
        real_prob = float(probs[real_idx])

        prediction = "fake" if fake_prob >= real_prob else "real"
        confidence = max(fake_prob, real_prob)

        return ModelOutput(
            prediction=prediction,
            confidence=confidence,
            probabilities={"real": real_prob, "fake": fake_prob},
            metadata={
                "backbone": self.backbone,
                "logits": logits.squeeze().cpu().numpy().tolist(),
                "class_to_idx": self._class_to_idx,
            },
            inference_time_ms=elapsed,
        )

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        return self._net(tensor)

    def get_explainability(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        """Generate Grad-CAM heatmap for prediction explanation."""
        tensor = self._preprocess(input_data)
        tensor.requires_grad_(True)

        # Forward pass
        output = self._net(tensor)
        pred_class = output.argmax(dim=-1).item()

        # Backward pass for gradients
        self._net.zero_grad()
        output[0, pred_class].backward()

        # Get gradients and feature maps
        if hasattr(self._net, 'features'):
            # Custom network
            features = self._net.features(tensor)
        else:
            # Pre-trained model — get last conv layer
            features = tensor  # Placeholder for actual hook

        # Generate heatmap
        heatmap = self._generate_gradcam_heatmap(features, tensor, pred_class)

        return {
            "heatmap": heatmap,
            "predicted_class": "fake" if pred_class == 1 else "real",
            "gradient_magnitude": float(features.grad.norm()) if features.grad is not None else 0.0,
        }

    def _generate_gradcam_heatmap(
        self,
        features: torch.Tensor,
        input_tensor: torch.Tensor,
        pred_class: int,
    ) -> np.ndarray:
        """Generate Grad-CAM visualization."""
        gradients = features.grad.data
        activations = features.data

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize to input size
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        return cam
