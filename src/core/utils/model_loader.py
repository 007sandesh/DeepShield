"""
Model loading and weight management utilities.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import torch
from loguru import logger


class ModelLoader:
    """
    Handles model weight loading with:
    - Weight file validation (checksum)
    - Automatic device placement
    - ONNX conversion support
    - Model caching
    """

    def __init__(self, weights_dir: str = "models/weights") -> None:
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def load_pytorch(
        self,
        model: torch.nn.Module,
        weights_path: str,
        device: str = "auto",
        strict: bool = True,
    ) -> torch.nn.Module:
        """Load PyTorch weights into a model."""
        path = Path(weights_path)

        if not path.exists():
            logger.warning(f"Weights not found: {path}")
            return model

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load weights
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict, strict=strict)

        logger.info(f"Loaded weights from {path.name}")
        return model

    def load_from_hub(
        self,
        model: torch.nn.Module,
        repo_id: str,
        filename: str = "model.pt",
        device: str = "auto",
    ) -> torch.nn.Module:
        """Load weights from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=repo_id, filename=filename)
            return self.load_pytorch(model, path, device)
        except ImportError:
            logger.error("huggingface_hub not installed")
            return model

    def convert_to_onnx(
        self,
        model: torch.nn.Module,
        output_path: str,
        input_shape: tuple = (1, 3, 256, 256),
    ) -> str:
        """Convert PyTorch model to ONNX format."""
        model.eval()
        dummy_input = torch.randn(input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        logger.info(f"ONNX model exported to {output_path}")
        return output_path

    def compute_checksum(self, file_path: str) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def verify_weights(self, file_path: str, expected_hash: Optional[str] = None) -> bool:
        """Verify weight file integrity."""
        if not Path(file_path).exists():
            return False

        if expected_hash is None:
            return True

        actual_hash = self.compute_checksum(file_path)
        return actual_hash == expected_hash
