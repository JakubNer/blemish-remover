"""Repeated blemish detection using OpenCV-based aggregation.

The implementation deliberately keeps detection lightweight and dependency-minimal:
it looks for small, persistent high-frequency artifacts that recur in the same
location across many images. This works well for dust spots, sensor blemishes,
and similar repeated defects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import cv2
import numpy as np

from src.utils import load_image, save_image

log = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Container for the detected blemish mask and preview metadata."""

    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float
    reference_shape: tuple[int, int]
    overlay_path: Path | None = None
    crop_path: Path | None = None
    representative_image: Path | None = None


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to a target ``(height, width)`` shape."""
    target_h, target_w = shape
    if mask.shape[:2] == (target_h, target_w):
        return (mask > 0).astype(np.uint8) * 255
    resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8) * 255


def _detail_map(image_bgr: np.ndarray) -> np.ndarray:
    """Extract a high-frequency residual map highlighting small blemishes."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=12.0)
    detail = cv2.absdiff(gray, background).astype(np.float32)
    detail = cv2.GaussianBlur(detail, (0, 0), sigmaX=1.0)

    max_value = float(detail.max())
    if max_value > 0:
        detail /= max_value
    return detail


def _choose_component(heatmap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pick the most likely blemish component from a thresholded heatmap mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = heatmap.shape
    image_area = h * w

    best_component: np.ndarray | None = None
    best_score = -1.0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        if area > max(4096, int(image_area * 0.02)):
            continue

        component = labels == label
        mean_score = float(heatmap[component].mean())
        area_bonus = min(1.75, 0.6 + area / 40.0)
        size_penalty = 1.0 + (area / max(64.0, image_area * 0.001))
        score = (mean_score * area_bonus) / size_penalty

        if score > best_score:
            best_score = score
            best_component = component

    if best_component is not None:
        return best_component.astype(np.uint8) * 255

    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    fallback = np.zeros_like(mask, dtype=np.uint8)
    radius = max(4, min(h, w) // 120)
    cv2.circle(fallback, (int(x), int(y)), radius, 255, -1)
    return fallback


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Detected mask is empty.")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def detect_common_blemish(image_paths: list[Path], cfg) -> DetectionResult:
    """Detect the repeated blemish shared across a batch of images."""
    if not image_paths:
        raise ValueError("No images were provided for blemish detection.")

    detail_maps: list[np.ndarray] = []
    reference_h = reference_w = 0

    for index, path in enumerate(image_paths):
        image = load_image(path)
        if index == 0:
            reference_h, reference_w = image.shape[:2]
        elif image.shape[:2] != (reference_h, reference_w):
            image = cv2.resize(image, (reference_w, reference_h), interpolation=cv2.INTER_AREA)

        detail_maps.append(_detail_map(image))

    stack = np.stack(detail_maps, axis=0)
    median_map = np.median(stack, axis=0)
    mean_map = np.mean(stack, axis=0)
    stability_map = 0.65 * median_map + 0.35 * mean_map

    quantile = min(0.999, max(0.975, 0.96 + 0.035 * float(cfg.match_threshold)))
    percentile_threshold = float(np.quantile(stability_map, quantile))
    ratio_threshold = float(stability_map.max()) * max(0.55, min(0.95, float(cfg.match_threshold)))
    threshold = max(percentile_threshold, ratio_threshold)

    binary = (stability_map >= threshold).astype(np.uint8) * 255
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large)

    mask = _choose_component(stability_map, binary)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)

    score = float(stability_map[mask > 0].mean()) if np.any(mask > 0) else 0.0
    bbox = _mask_bbox(mask)

    log.info(
        "Detected blemish with score %.4f at bbox x=%d y=%d w=%d h=%d",
        score,
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0] + 1,
        bbox[3] - bbox[1] + 1,
    )

    return DetectionResult(
        mask=mask.astype(np.uint8),
        bbox=bbox,
        score=score,
        reference_shape=(reference_h, reference_w),
        representative_image=image_paths[0],
    )


def save_detection_preview(
    image_path: Path,
    detection: DetectionResult,
    preview_dir: Path,
    padding: int = 48,
) -> DetectionResult:
    """Save preview images showing the detected blemish overlay and crop."""
    image = load_image(image_path)
    mask = resize_mask(detection.mask, image.shape[:2])
    x0, y0, x1, y1 = _mask_bbox(mask)

    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    overlay = np.where(mask[:, :, None] > 0, cv2.addWeighted(overlay, 0.35, red, 0.65, 0), overlay)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)

    crop_x0 = max(0, x0 - padding)
    crop_y0 = max(0, y0 - padding)
    crop_x1 = min(image.shape[1], x1 + padding)
    crop_y1 = min(image.shape[0], y1 + padding)
    crop = overlay[crop_y0:crop_y1, crop_x0:crop_x1].copy()

    preview_dir.mkdir(parents=True, exist_ok=True)
    base_name = image_path.stem
    overlay_path = preview_dir / f"{base_name}_blemish_overlay.png"
    crop_path = preview_dir / f"{base_name}_blemish_crop.png"

    save_image(overlay_path, overlay)
    save_image(crop_path, crop)

    detection.overlay_path = overlay_path
    detection.crop_path = crop_path
    detection.representative_image = image_path
    detection.bbox = (x0, y0, x1, y1)
    return detection