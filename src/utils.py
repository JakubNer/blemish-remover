"""Image I/O and utility helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image loading / saving
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load an image from disk as a BGR numpy array (OpenCV convention).

    Returns:
        H×W×3 uint8 array in BGR colour space.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image as an RGB numpy array."""
    img = load_image(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path: Path, img: np.ndarray) -> None:
    """Save a BGR numpy array to disk.

    Creates parent directories as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif ext == ".png":
        cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        cv2.imwrite(str(path), img)


def save_image_pil(path: Path, img: Image.Image) -> None:
    """Save a PIL Image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def discover_images(
    directory: Path,
    extensions: Iterable[str] | None = None,
    sample: int = 0,
) -> list[Path]:
    """Find all image files in *directory*.

    Parameters
    ----------
    directory : Path
        Root folder to scan.
    extensions : iterable of str, optional
        File extensions to consider (e.g. ``('.jpg', '.png')``).
    sample : int
        If > 0, return at most this many files (shuffled).

    Returns
    -------
    list[Path]
        Sorted list of image paths.
    """
    if extensions is None:
        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

    images: list[Path] = []
    for ext in extensions:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))

    # deduplicate & sort for deterministic ordering
    images = sorted(set(images))

    if sample and 0 < sample < len(images):
        import random
        random.seed(42)
        images = random.sample(images, sample)

    log.info("Discovered %d images in %s", len(images), directory)
    return images


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def progress_iter(iterable, desc: str = "", total: int | None = None):
    """Wrap an iterable in a tqdm progress bar."""
    return tqdm(iterable, desc=desc, total=total or len(iterable))


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def resize_to_fit(img: np.ndarray, max_dim: int = 2048) -> np.ndarray:
    """Resize image so the longest side is at most *max_dim*."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure the image is a 3-channel BGR array."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img