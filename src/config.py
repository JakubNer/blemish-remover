"""Configuration management for blemish-remover."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Holds all runtime configuration for the blemish remover."""

    # --- paths ---
    input_dir: Path = field(default_factory=lambda: Path("input"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    blemish_preview_dir: Path = field(default_factory=lambda: Path("blemish_previews"))

    # --- LaMa ---
    lama_venv_python: Optional[Path] = field(
        default_factory=lambda: Path("third_party/lama-runtime/.venv/Scripts/python.exe")
    )
    lama_enabled: bool = True

    # --- SDXL fallback ---
    sdxl_enabled: bool = True
    sdxl_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

    # --- detection ---
    # Number of images to sample for watermark detection (0 = all)
    detection_sample: int = 0
    # Detection strictness (0–1, higher = stricter)
    match_threshold: float = 0.58
    # Expected transparent watermark footprint used to bias component selection.
    expected_watermark_width: int = 2000
    expected_watermark_height: int = 1100
    # Largest side used for detection; images are downscaled only for analysis.
    detection_max_dim: int = 2560
    # Expand/merge detected watermark fragments before removal.
    mask_expand_px: int = 24
    # Extra context around the mask crop used for inpainting backends.
    inpaint_context_px: int = 160
    # Largest side fed into SDXL after cropping around the watermark.
    sdxl_max_side: int = 1536
    # Preview padding around the detected region.
    preview_padding_px: int = 160

    # --- general ---
    device: str = "cuda"  # "cuda" or "cpu"
    verbose: bool = False

    # --- image formats to consider ---
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

    # ------------------------------------------------------------------
    def resolve(self) -> None:
        """Validate and resolve paths."""
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        # Create output dirs if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.blemish_preview_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    @classmethod
    def from_cli_args(cls, args) -> "Config":
        """Build a Config from parsed argparse namespace."""
        cfg = cls()
        if hasattr(args, "input") and args.input:
            cfg.input_dir = Path(args.input)
        if hasattr(args, "output") and args.output:
            cfg.output_dir = Path(args.output)
        if hasattr(args, "device") and args.device:
            cfg.device = args.device
        if hasattr(args, "no_lama"):
            cfg.lama_enabled = not args.no_lama
        if hasattr(args, "no_sdxl"):
            cfg.sdxl_enabled = not args.no_sdxl
        if hasattr(args, "verbose"):
            cfg.verbose = args.verbose
        if hasattr(args, "threshold") and args.threshold is not None:
            cfg.match_threshold = args.threshold
        if hasattr(args, "sample") and args.sample is not None:
            cfg.detection_sample = args.sample
        return cfg