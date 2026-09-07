"""Main pipeline for the blemish remover CLI."""

from __future__ import annotations

from collections import Counter
import logging
import random

from src.detector import detect_common_blemish, localize_blemish_mask, save_detection_preview
from src.remover import BlemishRemover
from src.utils import discover_images, load_image, progress_iter, save_image

log = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def _select_detection_images(images: list, sample_size: int) -> list:
    if sample_size and 0 < sample_size < len(images):
        random.seed(42)
        return random.sample(images, sample_size)
    return images


def _confirm_detection() -> bool:
    while True:
        answer = input("Use this matched watermark mask for cleaning? [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def run_pipeline(cfg, skip_confirm: bool = False) -> None:
    """Run the full blemish detection and removal pipeline."""
    _configure_logging(cfg.verbose)

    images = discover_images(cfg.input_dir, extensions=cfg.image_extensions)
    if not images:
        raise RuntimeError(f"No supported images were found in {cfg.input_dir}")

    detection_images = _select_detection_images(images, cfg.detection_sample)
    log.info("Matching watermark guides against %d image(s)", len(detection_images))

    detection = detect_common_blemish(detection_images, cfg)
    preview_image = detection.representative_image or detection_images[0]
    detection = save_detection_preview(preview_image, detection, cfg.blemish_preview_dir)

    print("Matched watermark preview files:")
    if detection.overlay_path is not None:
        print(f"  Overlay: {detection.overlay_path}")
    if detection.crop_path is not None:
        print(f"  Crop:    {detection.crop_path}")
    print(f"  Score:   {detection.score:.4f}")

    if not skip_confirm and not _confirm_detection():
        raise RuntimeError("Blemish confirmation was declined by the user.")

    remover = BlemishRemover(cfg)
    summary: Counter[str] = Counter()

    for image_path in progress_iter(images, desc="Cleaning images"):
        image = load_image(image_path)
        image_mask, localization_score = localize_blemish_mask(image, detection, cfg)
        log.debug("Localized blemish in %s (score %.3f)", image_path.name, localization_score)
        cleaned, method = remover.remove_from_image(image, image_mask)
        save_image(cfg.output_dir / image_path.name, cleaned)
        summary[method] += 1

    print("\nFinished cleaning images.")
    print(f"Output directory: {cfg.output_dir}")
    print("Methods used:")
    for method, count in sorted(summary.items()):
        print(f"  {method}: {count}")