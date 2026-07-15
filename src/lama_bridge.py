"""Bridge to LaMa inference running inside its dedicated virtual environment."""

from __future__ import annotations

from pathlib import Path
import logging
import subprocess
import sys
import tempfile
import textwrap

import cv2
import numpy as np

from src.utils import save_image

log = logging.getLogger(__name__)


class LamaBridge:
    """Run simple-lama-inpainting via the separate LaMa virtual environment."""

    def __init__(self, python_executable: Path | None) -> None:
        self.python_executable = Path(python_executable) if python_executable else None
        self._availability_checked = False
        self._is_available = False

    def is_available(self) -> bool:
        """Return whether the configured LaMa environment looks usable."""
        if self._availability_checked:
            return self._is_available

        self._availability_checked = True
        if self.python_executable is None or not self.python_executable.exists():
            log.info("LaMa runtime not found at %s", self.python_executable)
            self._is_available = False
            return False

        command = [
            str(self.python_executable),
            "-c",
            "from simple_lama_inpainting import SimpleLama; print('ok')",
        ]

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            self._is_available = True
        except Exception as exc:
            log.warning("LaMa runtime check failed: %s", exc)
            self._is_available = False

        return self._is_available

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint an image using LaMa in its dedicated environment."""
        if not self.is_available():
            raise RuntimeError("LaMa runtime is not available.")

        mask = (mask > 0).astype(np.uint8) * 255

        with tempfile.TemporaryDirectory(prefix="blemish_lama_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "input.png"
            mask_path = tmp_path / "mask.png"
            output_path = tmp_path / "output.png"

            save_image(image_path, image_bgr)
            cv2.imwrite(str(mask_path), mask)

            script = textwrap.dedent(
                """
                from pathlib import Path
                import sys
                import numpy as np
                from PIL import Image
                from simple_lama_inpainting import SimpleLama

                image_path = Path(sys.argv[1])
                mask_path = Path(sys.argv[2])
                output_path = Path(sys.argv[3])

                image = Image.open(image_path).convert('RGB')
                mask = Image.open(mask_path).convert('L')
                lama = SimpleLama()
                result = lama(image, mask)
                result.save(output_path)
                """
            ).strip()

            command = [
                str(self.python_executable),
                "-c",
                script,
                str(image_path),
                str(mask_path),
                str(output_path),
            ]

            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            if completed.stdout.strip():
                log.debug("LaMa stdout: %s", completed.stdout.strip())
            if completed.stderr.strip():
                log.debug("LaMa stderr: %s", completed.stderr.strip())

            result = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            if result is None:
                raise RuntimeError("LaMa completed but did not produce a readable output image.")
            return result