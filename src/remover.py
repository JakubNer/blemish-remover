"""Blemish removal orchestration.

Removal order:
1. LaMa inpainting, if available and enabled.
2. SDXL inpainting, if available and enabled.
3. OpenCV Telea inpainting as a robust local fallback.
"""

from __future__ import annotations

from collections import Counter
import logging

import cv2
import numpy as np

from src.detector import resize_mask
from src.lama_bridge import LamaBridge
from src.sdxl_inpainter import SDXLInpainter

log = logging.getLogger(__name__)


class BlemishRemover:
    """Remove a detected blemish from images using configured backends."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.lama = LamaBridge(cfg.lama_venv_python) if cfg.lama_enabled else None
        self.sdxl = SDXLInpainter(cfg.sdxl_model_id, device=cfg.device) if cfg.sdxl_enabled else None

    def remove_from_image(self, image_bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, str]:
        """Remove the blemish from a single image and return the method used."""
        scaled_mask = resize_mask(mask, image_bgr.shape[:2])
        if not np.any(scaled_mask > 0):
            return image_bgr.copy(), "noop"

        if self.lama is not None and self.lama.is_available():
            try:
                return self.lama.inpaint(image_bgr, scaled_mask), "lama"
            except Exception as exc:
                log.warning("LaMa inpainting failed, falling back: %s", exc)

        if self.sdxl is not None:
            try:
                return self.sdxl.inpaint(image_bgr, scaled_mask), "sdxl"
            except Exception as exc:
                log.warning("SDXL inpainting failed, falling back: %s", exc)

        fallback = cv2.inpaint(image_bgr, scaled_mask, 3, cv2.INPAINT_TELEA)
        return fallback, "opencv"


def init_summary_counter() -> Counter:
    """Create a summary counter for processed outputs."""
    return Counter()