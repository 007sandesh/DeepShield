"""
GPU and device management utilities.

Handles device selection, memory management, and mixed precision.
"""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from typing import Optional

import torch
from loguru import logger


class GPUManager:
    """
    Centralized GPU/device management.

    Features:
    - Auto-detect best device
    - Memory monitoring and cleanup
    - Mixed precision context manager
    - Multi-GPU support
    """

    def __init__(self, device: str = "auto") -> None:
        self.device = self._resolve_device(device)
        self._is_cuda = self.device.type == "cuda"

        if self._is_cuda:
            logger.info(
                f"GPU: {torch.cuda.get_device_name(self.device)} "
                f"({self._get_total_memory() / 1e9:.1f} GB)"
            )
        else:
            logger.info("Running on CPU")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")

        return torch.device(device)

    def _get_total_memory(self) -> int:
        """Get total GPU memory in bytes."""
        if self._is_cuda:
            return torch.cuda.get_device_properties(self.device).total_mem
        return 0

    def get_memory_usage(self) -> dict[str, float]:
        """Get current GPU memory usage."""
        if not self._is_cuda:
            return {"allocated_mb": 0, "cached_mb": 0, "total_mb": 0}

        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / 1e6,
            "cached_mb": torch.cuda.memory_reserved(self.device) / 1e6,
            "total_mb": self._get_total_memory() / 1e6,
        }

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self._is_cuda:
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("GPU cache cleared")

    def move_to_device(self, tensor_or_model):
        """Move tensor or model to the managed device."""
        if isinstance(tensor_or_model, torch.Tensor):
            return tensor_or_model.to(self.device)
        elif isinstance(tensor_or_model, torch.nn.Module):
            return tensor_or_model.to(self.device)
        return tensor_or_model

    @contextmanager
    def mixed_precision(self, enabled: bool = True):
        """Context manager for mixed precision inference."""
        if enabled and self._is_cuda:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                yield
        else:
            yield

    @contextmanager
    def no_grad(self):
        """Context manager for inference (no gradient computation)."""
        with torch.no_grad():
            yield


def get_device(device: str = "auto") -> torch.device:
    """Quick device resolver."""
    return GPUManager._resolve_device(device)
