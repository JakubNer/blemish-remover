"""Guide-based detection for large semi-transparent watermarks.

User-provided rough artwork defines the complete watermark shape. OpenCV then
matches that guide independently in each photograph over position and scale.
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
    """Container for the guide-based watermark model and preview metadata."""

    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float
    reference_shape: tuple[int, int]
    response_template: np.ndarray | None = None
    detection_shape: tuple[int, int] | None = None
    guide_templates: list[tuple[np.ndarray, tuple[float, float]]] | None = None
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
    """Build a response map for a broad, light semi-transparent watermark.

    Signed bright residuals are intentional: ordinary dark scene edges used to
    be mistaken for the logo. A white watermark remains brighter than its local
    background at several scales, including its fine URL text and broad GP.
    """
    image_float = image_bgr.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image_float, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.8)

    residuals = []
    for sigma in (2.0, 5.0, 12.0, 25.0, 55.0):
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        residuals.append(np.maximum(gray - background, 0.0))

    response = np.maximum.reduce(residuals).astype(np.float32)
    robust_high = float(np.quantile(response, 0.995))
    if robust_high > 0:
        response = np.clip(response / robust_high, 0.0, 1.0)
    return cv2.GaussianBlur(response, (0, 0), sigmaX=1.0)


def _aggregate_repeated_response(response_maps: list[np.ndarray], cfg) -> np.ndarray:
    """Keep responses present in every image while tolerating small drift.

    Local max filtering permits slight shifts. A low cross-image percentile
    acts like a softened intersection, strongly rejecting motorcycle and road
    detail that occurs in only one photograph.
    """
    if not response_maps:
        raise ValueError("At least one response map is required.")
    h, w = response_maps[0].shape
    radius = max(1, int(round(min(h, w) * float(cfg.aggregation_tolerance_ratio))))
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tolerant_maps = [cv2.dilate(response.astype(np.float32), kernel) for response in response_maps]
    stack = np.stack(tolerant_maps, axis=0)
    repeated = np.quantile(stack, 0.15, axis=0).astype(np.float32)
    consensus = repeated * (0.65 + 0.35 * np.mean(stack, axis=0))
    return _normalize_map(consensus)


def _expected_priors(cfg, shape: tuple[int, int]) -> tuple[float, float]:
    # A ratio is resolution independent. The repeated artifact is known to
    # occupy at least one tenth of the image, while the configured dimensions
    # still provide a useful aspect-ratio prior.
    expected_area_ratio = max(0.10, float(cfg.min_blemish_area_ratio))
    expected_area_ratio = min(0.90, expected_area_ratio)
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


def _guide_mask(path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    """Extract artwork from an RGBA guide or a dark-on-light rough sketch."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Could not read watermark guide: {path}")

    if raw.ndim == 3 and raw.shape[2] == 4 and np.any(raw[:, :, 3] < 255):
        strength = raw[:, :, 3]
    else:
        bgr = raw[:, :, :3] if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        background = float(np.median(border))
        strength = np.clip(np.abs(gray.astype(np.float32) - background), 0, 255).astype(np.uint8)

    nonzero = strength[strength > 2]
    if nonzero.size == 0:
        raise RuntimeError(f"Watermark guide has no visible artwork: {path}")
    otsu, _ = cv2.threshold(strength, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(5, min(32, int(round(otsu * 0.40))))
    mask = (strength >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    minimum = max(4, int(mask.size * 0.000003))
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum:
            cleaned[labels == label] = 255

    x0, y0, x1, y1 = _mask_bbox(cleaned)
    footprint = ((x1 - x0 + 1) / raw.shape[1], (y1 - y0 + 1) / raw.shape[0])
    return cleaned[y0:y1 + 1, x0:x1 + 1], footprint


def _load_guide_templates(cfg) -> list[tuple[np.ndarray, tuple[float, float]]]:
    paths = sorted(
        path for path in cfg.watermark_dir.iterdir()
        if path.is_file() and path.suffix.lower() in cfg.image_extensions
    )
    if not paths:
        raise RuntimeError(f"No watermark guide images were found in {cfg.watermark_dir}")
    templates = [_guide_mask(path) for path in paths]
    log.info("Loaded %d watermark guide(s) from %s", len(templates), cfg.watermark_dir)
    return templates


def _match_guide(
    image_bgr: np.ndarray,
    guides: list[tuple[np.ndarray, tuple[float, float]]],
    cfg,
) -> tuple[np.ndarray, float]:
    """Match rough guide artwork against one image over position and scale."""
    detection_image = _resize_for_detection(image_bgr, cfg.detection_max_dim)
    h, w = detection_image.shape[:2]
    response = _watermark_response_map(detection_image)
    best_score = -1.0
    best_mask: np.ndarray | None = None
    steps = max(3, int(cfg.scale_search_steps))
    scales = np.linspace(
        max(0.45, 1.0 - float(cfg.guide_scale_tolerance_ratio)),
        1.0 + float(cfg.guide_scale_tolerance_ratio),
        steps,
    )

    for guide_mask, (width_ratio, height_ratio) in guides:
        base_w = max(12, int(round(w * width_ratio)))
        base_h = max(12, int(round(h * height_ratio)))
        for scale in scales:
            candidate_w = int(round(base_w * scale))
            candidate_h = int(round(base_h * scale))
            if candidate_w < 8 or candidate_h < 8 or candidate_w > w or candidate_h > h:
                continue
            candidate_mask = cv2.resize(
                guide_mask, (candidate_w, candidate_h), interpolation=cv2.INTER_LINEAR
            )
            candidate = cv2.GaussianBlur(candidate_mask.astype(np.float32) / 255.0, (0, 0), 1.2)
            scores = cv2.matchTemplate(response, candidate, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(scores)
            if math.isfinite(score) and score > best_score:
                best_score = float(score)
                best_mask = _place_scaled_mask(
                    (candidate_mask >= 20).astype(np.uint8) * 255,
                    (h, w),
                    location,
                    (candidate_w, candidate_h),
                )

    if best_mask is None:
        raise RuntimeError("The watermark guides could not be matched to the image.")
    expand = max(1, int(round(cfg.mask_expand_px * min(h, w) / max(1, cfg.detection_max_dim))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1))
    best_mask = cv2.dilate(best_mask, kernel)
    return resize_mask(best_mask, image_bgr.shape[:2]), best_score


def detect_common_blemish(image_paths: list[Path], cfg) -> DetectionResult:
    """Build a guide-based detector and locate it in a representative image."""
    if not image_paths:
        raise ValueError("No images were provided for watermark detection.")
    guides = _load_guide_templates(cfg)
    best: tuple[float, Path, np.ndarray, tuple[int, int]] | None = None
    for path in image_paths:
        image = load_image(path)
        mask, score = _match_guide(image, guides, cfg)
        if best is None or score > best[0]:
            best = (score, path, mask, image.shape[:2])

    if best is None:
        raise RuntimeError("No input image could be matched against the watermark guides.")
    score, representative, mask, reference_shape = best
    bbox = _mask_bbox(mask)
    log.info(
        "Matched watermark guide with score %.4f at bbox x=%d y=%d w=%d h=%d",
        score, bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1,
    )
    return DetectionResult(
        mask=mask,
        bbox=bbox,
        score=score,
        reference_shape=reference_shape,
        guide_templates=guides,
        representative_image=representative,
    )


def _place_scaled_mask(
    source_mask: np.ndarray,
    target_shape: tuple[int, int],
    top_left: tuple[int, int],
    scaled_size: tuple[int, int],
) -> np.ndarray:
    """Resize a template mask and place it in a target detection canvas."""
    target_h, target_w = target_shape
    scaled_w, scaled_h = scaled_size
    scaled = cv2.resize(source_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    x, y = top_left
    output = np.zeros((target_h, target_w), dtype=np.uint8)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(target_w, x + scaled_w), min(target_h, y + scaled_h)
    if x1 > x0 and y1 > y0:
        output[y0:y1, x0:x1] = scaled[y0 - y:y1 - y, x0 - x:x1 - x]
    return output


def localize_blemish_mask(
    image_bgr: np.ndarray,
    detection: DetectionResult,
    cfg,
) -> tuple[np.ndarray, float]:
    """Match the user-provided rough watermark guide in one image."""
    if detection.guide_templates:
        mask, score = _match_guide(image_bgr, detection.guide_templates, cfg)
        if score < float(cfg.localization_min_score):
            log.warning("Watermark guide match score is low: %.3f", score)
        return mask, score
    return resize_mask(detection.mask, image_bgr.shape[:2]), detection.score


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