"""
Vision Transformer Attention Network for patch-level analysis.

Uses self-attention maps to detect:
- Local inconsistencies at patch boundaries
- Attention pattern anomalies
- Manipulation boundary detection
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.core.models.base import BaseModel, ModelOutput, ModelType, ModelRegistry


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, img_size: int = 256, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with multi-head self-attention."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with attention weights
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


class AttentionAnalysisNet(nn.Module):
    """Vision Transformer for attention-based deepfake detection."""

    def __init__(self, img_size: int = 256, patch_size: int = 16, embed_dim: int = 768, num_heads: int = 12, num_layers: int = 6) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        attention_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attention_maps.append(attn)

        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)

        return logits, attention_maps


@ModelRegistry.register
class AttentionNetwork(BaseModel):
    """
    Vision Transformer attention analysis for deepfake detection.

    Uses self-attention patterns to identify:
    1. Patch-level inconsistencies
    2. Attention distribution anomalies
    3. Manipulation boundary localization
    4. Cross-attention pattern analysis
    """

    model_type = ModelType.ATTENTION
    model_name = "attention_network"

    def __init__(self, device: str = "auto", num_layers: int = 6, **kwargs) -> None:
        super().__init__(device=device)
        self._net = AttentionAnalysisNet(num_layers=num_layers)
        self._net = self._net.to(self.device).eval()
        self._is_loaded = True
        logger.info(f"Attention network initialized ({num_layers} layers)")

    def load_weights(self, weights_path: Optional[str] = None) -> None:
        """Load attention network weights."""
        if weights_path:
            state_dict = torch.load(weights_path, map_location=self.device)
            self._net.load_state_dict(state_dict)
        self._is_loaded = True

    def predict(self, input_data: np.ndarray, **kwargs) -> ModelOutput:
        """
        Analyze attention patterns for forgery detection.

        Args:
            input_data: Face image (H, W, 3)

        Returns:
            ModelOutput with attention analysis
        """
        tensor = self._preprocess(input_data)

        result, elapsed = self._measure_inference(self._forward, tensor)

        logits, attention_maps = result
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Analyze attention patterns
        attn_analysis = self._analyze_attention_patterns(attention_maps)

        return ModelOutput(
            prediction="fake" if probs[1] > 0.5 else "real",
            confidence=float(max(probs[0], probs[1])),
            probabilities={"real": float(probs[0]), "fake": float(probs[1])},
            metadata={
                "attention_entropy": attn_analysis["entropy"],
                "attention_uniformity": attn_analysis["uniformity"],
                "boundary_score": attn_analysis["boundary_score"],
                "num_anomalous_patches": attn_analysis["num_anomalous"],
            },
            explainability={
                "attention_maps": [a.detach().cpu().numpy() for a in attention_maps],
                "patch_scores": attn_analysis["patch_scores"],
                "manipulation_map": attn_analysis["manipulation_map"],
            },
            inference_time_ms=elapsed,
        )

    @torch.no_grad()
    def _forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass returning logits and attention maps."""
        return self._net(tensor)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to model input tensor."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self.device)

    def _analyze_attention_patterns(self, attention_maps: list[torch.Tensor]) -> dict[str, Any]:
        """Analyze attention patterns for anomalies."""
        # MultiheadAttention returns (B, L, S) when average_attn_weights=True (default),
        # or (B, num_heads, L, S) when average_attn_weights=False.
        last_attn = attention_maps[-1]
        if last_attn.dim() == 4:
            attn_np = last_attn[0, :, 0, 1:].detach().cpu().numpy()
            patch_scores = attn_np.mean(axis=0)
        elif last_attn.dim() == 3:
            patch_scores = last_attn[0, 0, 1:].detach().cpu().numpy()
            attn_np = patch_scores[np.newaxis, :]
        else:
            patch_scores = np.zeros(1, dtype=np.float32)
            attn_np = patch_scores[np.newaxis, :]

        # Attention entropy (uniformity measure)
        attn_probs = np.exp(attn_np) / (np.exp(attn_np).sum(axis=-1, keepdims=True) + 1e-8)
        entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-8), axis=-1)
        mean_entropy = float(np.mean(entropy))

        # Attention uniformity (high = suspicious for deepfakes)
        uniformity = float(np.std(attn_np))

        num_patches = len(patch_scores)
        grid_size = int(np.sqrt(num_patches))

        # Reshape to spatial grid
        if grid_size * grid_size == num_patches and num_patches > 0:
            spatial_scores = patch_scores.reshape(grid_size, grid_size).astype(np.float64)
            spatial_scores = cv2.resize(spatial_scores, (256, 256))
        else:
            spatial_scores = np.zeros((256, 256), dtype=np.float64)

        # Detect manipulation boundaries (high gradient in attention)
        grad_x = cv2.Sobel(spatial_scores, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(spatial_scores, cv2.CV_64F, 0, 1, ksize=3)
        boundary_map = np.sqrt(grad_x ** 2 + grad_y ** 2)
        boundary_score = float(np.mean(boundary_map))

        # Count anomalous patches (attention > 2 standard deviations)
        if len(patch_scores) > 0:
            threshold = float(np.mean(patch_scores) + 2 * np.std(patch_scores))
            num_anomalous = int(np.sum(patch_scores > threshold))
        else:
            num_anomalous = 0

        return {
            "entropy": mean_entropy,
            "uniformity": uniformity,
            "boundary_score": boundary_score,
            "num_anomalous": num_anomalous,
            "patch_scores": patch_scores.tolist(),
            "manipulation_map": spatial_scores,
        }

    def get_explainability(self, input_data: np.ndarray, **kwargs) -> dict[str, Any]:
        """Generate attention visualization."""
        output = self.predict(input_data, **kwargs)
        return {
            "attention_maps": output.explainability["attention_maps"],
            "manipulation_map": output.explainability["manipulation_map"],
            "patch_scores": output.explainability["patch_scores"],
            "attention_entropy": output.metadata["attention_entropy"],
            "boundary_score": output.metadata["boundary_score"],
        }
