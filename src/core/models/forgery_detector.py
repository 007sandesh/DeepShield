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
    Deepfake detection using fine-tuned vision backbone.

    Supports:
    - EfficientNet-B4 with DeepShield Celeb-DF / FF++ weights (default)
    - Custom XceptionNet fallback
    - HuggingFace ViT
    """

    model_type = ModelType.FORGERY_DETECTION
    model_name = "forgery_detector"

    DEFAULT_WEIGHT_PATHS = {
        "efficientnet": "models/weights/efficientnet_celebdf.pt",
        "xception": "models/weights/xception_ffpp.pt",
        "vit": "models/weights/vit_face_forensics.pt",
    }

    def __init__(
        self,
        backbone: str = "efficientnet",
        device: str = "auto",
        use_pretrained: bool = True,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__(device=device)
        self.backbone = backbone
        self.use_pretrained = use_pretrained
        self._weights_path = weights_path
        self._net = None
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
        if weights_path is None:
            default = Path(self.DEFAULT_WEIGHT_PATHS.get(self.backbone, ""))
            if default.exists():
                weights_path = str(default)

        if self.backbone == "efficientnet":
            self._net = self._build_efficientnet()
            self._input_size = 380
            if weights_path:
                self._load_efficientnet_checkpoint(weights_path)
            elif self.use_pretrained:
                logger.warning(
                    "EfficientNet deepfake weights missing; using ImageNet features only"
                )

        elif self.backbone == "xception":
            self._net = ForgeryDetectorNet(num_classes=2)
            self._input_size = 256
            if weights_path:
                loaded = self._try_load_state_dict(weights_path, strict=False)
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
            if weights_path:
                self._try_load_state_dict(weights_path, strict=False)
            elif self.use_pretrained:
                from transformers import ViTModel

                pretrained = ViTModel.from_pretrained("google/vit-base-patch16-224")
                self._net.vit.load_state_dict(pretrained.state_dict(), strict=False)

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        self._net = self._net.to(self.device).eval()
        self._transform_params = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        self._is_loaded = True
        self._load_time_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Forgery detector ({self.backbone}) loaded in {self._load_time_ms:.1f}ms "
            f"[classes={self._class_to_idx}]"
        )

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

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR image to normalized tensor."""
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size = self._input_size
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LANCZOS4)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor(self._transform_params["mean"]).view(3, 1, 1)
        std = torch.tensor(self._transform_params["std"]).view(3, 1, 1)
        tensor = (tensor - mean) / std
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
