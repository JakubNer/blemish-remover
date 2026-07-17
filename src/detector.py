"""Repeated watermark detection using OpenCV-based aggregation.

This module is tuned for large, semi-transparent watermarks that recur in the
same location across many images. Unlike the original blemish-focused logic,
it supports broad masks, multiple disconnected watermark fragments, and a size
prior for watermark footprints around 2000x1100 pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import math

import cv2
import numpy as np

from src.utils import load_image, save_image

log = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Container for the detected watermark mask and preview metadata."""

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


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def _resize_for_detection(image_bgr: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if max(h, w) <= max_dim:
        return image_bgr
    scale = max_dim / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _watermark_response_map(image_bgr: np.ndarray) -> np.ndarray:
    """Build a response map for broad transparent watermarks."""
    image_float = image_bgr.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image_float, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_float, cv2.COLOR_BGR2HSV)

    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    gray_blur_medium = cv2.GaussianBlur(gray, (0, 0), sigmaX=24.0)
    gray_blur_large = cv2.GaussianBlur(gray, (0, 0), sigmaX=72.0)

    detail_medium = np.abs(gray - gray_blur_medium)
    detail_large = np.abs(gray - gray_blur_large)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    saturation_residual = np.abs(saturation - cv2.GaussianBlur(saturation, (0, 0), sigmaX=24.0))
    value_residual = np.abs(value - cv2.GaussianBlur(value, (0, 0), sigmaX=24.0))

    response = (
        0.36 * _normalize_map(detail_medium)
        + 0.24 * _normalize_map(detail_large)
        + 0.22 * _normalize_map(gradient)
        + 0.10 * _normalize_map(saturation_residual)
        + 0.08 * _normalize_map(value_residual)
    )
    response = cv2.GaussianBlur(response.astype(np.float32), (0, 0), sigmaX=2.0)
    return _normalize_map(response)


def _expected_priors(cfg, shape: tuple[int, int]) -> tuple[float, float]:
    h, w = shape
    expected_area = max(1.0, float(cfg.expected_watermark_width) * float(cfg.expected_watermark_height))
    expected_area_ratio = min(0.95, expected_area / max(1.0, float(h * w)))
    expected_aspect = float(cfg.expected_watermark_width) / max(1.0, float(cfg.expected_watermark_height))
    return expected_area_ratio, expected_aspect


def _threshold_heatmap(heatmap: np.ndarray, cfg) -> np.ndarray:
    strictness = float(np.clip(cfg.match_threshold, 0.05, 0.95))
    quantile = 0.84 + 0.11 * strictness
    percentile_threshold = float(np.quantile(heatmap, min(0.995, quantile)))
    ratio_threshold = float(heatmap.max()) * (0.42 + 0.30 * strictness)
    threshold = max(percentile_threshold, ratio_threshold)

    binary = (heatmap >= threshold).astype(np.uint8) * 255
    base = max(3, int(round(min(heatmap.shape[:2]) * 0.006)))
    if base % 2 == 0:
        base += 1
    close_size = max(base, int(round(min(heatmap.shape[:2]) * 0.018)))
    if close_size % 2 == 0:
        close_size += 1

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base, base))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    return binary


def _select_watermark_mask(heatmap: np.ndarray, binary: np.ndarray, cfg) -> np.ndarray:
    """Keep multiple strong components instead of a single tiny blemish."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = heatmap.shape
    image_area = float(h * w)
    expected_area_ratio, expected_aspect = _expected_priors(cfg, heatmap.shape)

    scored_components: list[tuple[float, np.ndarray]] = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(32, int(image_area * 0.0002)):
            continue
        if area > int(image_area * 0.92):
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue

        component = labels == label
        mean_score = float(heatmap[component].mean())
        area_ratio = area / image_area
        bbox_area_ratio = (bw * bh) / image_area
        aspect = bw / max(1.0, float(bh))

        area_prior = math.exp(-abs(math.log(max(area_ratio, 1e-6) / max(expected_area_ratio, 1e-6))))
        aspect_prior = math.exp(-abs(math.log(max(aspect, 1e-6) / max(expected_aspect, 1e-6))))
        bbox_prior = math.exp(-abs(math.log(max(bbox_area_ratio, 1e-6) / max(expected_area_ratio, 1e-6))))
        fill_ratio = area / max(1.0, float(bw * bh))
        fill_bonus = 0.65 + min(0.55, fill_ratio)

        score = mean_score * (0.45 + 0.75 * area_prior + 0.35 * aspect_prior + 0.25 * bbox_prior) * fill_bonus
        scored_components.append((score, component))

    if not scored_components:
        threshold = float(np.quantile(heatmap, 0.92))
        return (heatmap >= threshold).astype(np.uint8) * 255

    best_score = max(score for score, _ in scored_components)
    keep_threshold = best_score * 0.55
    selected = np.zeros_like(binary, dtype=np.uint8)
    for score, component in scored_components:
        if score >= keep_threshold:
            selected[component] = 255

    join_kernel_size = max(5, int(round(min(h, w) * 0.012)))
    if join_kernel_size % 2 == 0:
        join_kernel_size += 1
    join_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (join_kernel_size, join_kernel_size))
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, join_kernel)

    expand = max(1, int(round(cfg.mask_expand_px * min(h, w) / max(1, cfg.detection_max_dim))))
    expand_kernel_size = max(3, 2 * expand + 1)
    expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_kernel_size, expand_kernel_size))
    selected = cv2.dilate(selected, expand_kernel, iterations=1)
    return selected


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Detected mask is empty.")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def detect_common_blemish(image_paths: list[Path], cfg) -> DetectionResult:
    """Detect a repeated transparent watermark shared across a batch of images."""
    if not image_paths:
        raise ValueError("No images were provided for watermark detection.")

    response_maps: list[np.ndarray] = []
    reference_h = reference_w = 0
    detection_h = detection_w = 0

    for index, path in enumerate(image_paths):
        image = load_image(path)
        if index == 0:
            reference_h, reference_w = image.shape[:2]
            detection_image = _resize_for_detection(image, cfg.detection_max_dim)
            detection_h, detection_w = detection_image.shape[:2]
        else:
            if image.shape[:2] != (reference_h, reference_w):
                image = cv2.resize(image, (reference_w, reference_h), interpolation=cv2.INTER_AREA)
            detection_image = cv2.resize(image, (detection_w, detection_h), interpolation=cv2.INTER_AREA)

        response_maps.append(_watermark_response_map(detection_image))

    stack = np.stack(response_maps, axis=0)
    median_map = np.median(stack, axis=0)
    mean_map = np.mean(stack, axis=0)
    stability_map = _normalize_map(0.72 * median_map + 0.28 * mean_map)

    binary = _threshold_heatmap(stability_map, cfg)
    mask = _select_watermark_mask(stability_map, binary, cfg)
    mask = resize_mask(mask, (reference_h, reference_w))

    score = float(stability_map[mask.resize((detection_h, detection_w)) > 0].mean()) if False else 0.0
    bbox = _mask_bbox(mask)

    detection_view_mask = resize_mask(mask, (detection_h, detection_w))
    score = float(stability_map[detection_view_mask > 0].mean()) if np.any(detection_view_mask > 0) else 0.0

    log.info(
        "Detected watermark with score %.4f at bbox x=%d y=%d w=%d h=%d",
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
    """Save preview images showing the detected watermark overlay and crop."""
    image = load_image(image_path)
    mask = resize_mask(detection.mask, image.shape[:2])
    x0, y0, x1, y1 = _mask_bbox(mask)

    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    overlay = np.where(mask[:, :, None] > 0, cv2.addWeighted(overlay, 0.45, red, 0.55, 0), overlay)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)

    dynamic_padding = max(padding, int(round(max(x1 - x0 + 1, y1 - y0 + 1) * 0.08)))
    crop_x0 = max(0, x0 - dynamic_padding)
    crop_y0 = max(0, y0 - dynamic_padding)
    crop_x1 = min(image.shape[1], x1 + dynamic_padding)
    crop_y1 = min(image.shape[0], y1 + dynamic_padding)
    crop = overlay[crop_y0:crop_y1, crop_x0:crop_x1].copy()

    preview_dir.mkdir(parents=True, exist_ok=True)
    base_name = image_path.stem
    overlay_path = preview_dir / f"{base_name}_watermark_overlay.png"
    crop_path = preview_dir / f"{base_name}_watermark_crop.png"

    save_image(overlay_path, overlay)
    save_image(crop_path, crop)

    detection.overlay_path = overlay_path
    detection.crop_path = crop_path
    detection.representative_image = image_path
    detection.bbox = (x0, y0, x1, y1)
    return detection