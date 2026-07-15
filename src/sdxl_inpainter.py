"""SDXL inpainting fallback implementation."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class SDXLInpainter:
    """Lazy-loading SDXL inpainting wrapper using diffusers."""

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        from diffusers import AutoPipelineForInpainting

        dtype = torch.float16 if self.device == "cuda" and torch.cuda.is_available() else torch.float32
        extra_kwargs = {"torch_dtype": dtype}
        if dtype == torch.float16:
            extra_kwargs["variant"] = "fp16"

        log.info("Loading SDXL inpainting model: %s", self.model_id)
        pipe = AutoPipelineForInpainting.from_pretrained(self.model_id, **extra_kwargs)
        pipe = pipe.to(self.device if self.device == "cuda" and torch.cuda.is_available() else "cpu")
        self._pipe = pipe
        return pipe

    def inpaint(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        prompt: str = "clean natural photo background, remove dust spot, preserve image content",
        negative_prompt: str = "distortion, blur, duplicated objects, artifacts, smearing",
    ) -> np.ndarray:
        """Inpaint an image using an SDXL inpainting pipeline."""
        pipe = self._load()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=24,
            guidance_scale=6.5,
            strength=0.99,
        ).images[0]

        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)