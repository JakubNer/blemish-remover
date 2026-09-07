"""Tests for rough-guide watermark detection and localization."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.config import Config
from src.detector import DetectionResult, _guide_mask, _match_guide, localize_blemish_mask


class GuidedLocalizationTests(unittest.TestCase):
    def test_extracts_dark_artwork_from_light_jpeg_guide(self) -> None:
        image = np.full((100, 160, 3), 255, dtype=np.uint8)
        cv2.putText(image, "GP", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (90, 90, 90), 5)
        cv2.rectangle(image, (20, 75), (140, 88), (110, 110, 110), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.jpg"
            cv2.imwrite(str(path), image)
            mask, footprint = _guide_mask(path)

        self.assertGreater(int(np.count_nonzero(mask)), 400)
        self.assertGreater(footprint[0], 0.65)
        self.assertGreater(footprint[1], 0.55)

    def test_matches_rough_shape_at_arbitrary_location(self) -> None:
        shape = (180, 260)
        guide = np.zeros((45, 70), dtype=np.uint8)
        cv2.putText(guide, "GP", (2, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
        cv2.rectangle(guide, (3, 39), (67, 44), 255, 1)
        footprint = (0.30, 0.32)
        candidate = cv2.resize(guide, (78, 58), interpolation=cv2.INTER_LINEAR)
        response = np.zeros(shape, dtype=np.float32)
        response[73:131, 121:199] = cv2.GaussianBlur(
            candidate.astype(np.float32) / 255.0, (0, 0), 1.2
        )

        cfg = Config()
        cfg.guide_scale_tolerance_ratio = 0.0
        cfg.scale_search_steps = 3
        cfg.mask_expand_px = 0
        image = np.zeros((*shape, 3), dtype=np.uint8)

        with patch("src.detector._watermark_response_map", return_value=response):
            localized, score = _match_guide(image, [(guide, footprint)], cfg)

        ys, xs = np.where(localized > 0)
        self.assertGreater(score, 0.90)
        self.assertAlmostEqual(int(xs.min()), 121, delta=3)
        self.assertAlmostEqual(int(ys.min()), 85, delta=3)
        self.assertGreater(int(xs.max()), 190)
        self.assertGreater(int(ys.max()), 124)

    def test_legacy_result_without_guides_uses_batch_mask(self) -> None:
        shape = (100, 120)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[20:60, 30:80] = 255
        detection = DetectionResult(mask, (30, 20, 79, 59), 0.7, shape)
        localized, _ = localize_blemish_mask(
            np.zeros((*shape, 3), dtype=np.uint8), detection, Config()
        )
        np.testing.assert_array_equal(localized, mask)


if __name__ == "__main__":
    unittest.main()
