"""CLI entry point for blemish-remover."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="blemish-remover",
        description="Find a guided watermark in each image and remove it.",
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        default="input",
        help="Path to folder containing images with the blemish (default: ./input)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output",
        help="Path to folder for cleaned images (default: ./output)",
    )
    parser.add_argument(
        "-w", "--watermark",
        type=str,
        default="watermark",
        help="Folder containing rough watermark guide images (default: ./watermark)",
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--no-lama",
        action="store_true",
        help="Disable LaMa inpainting (use only SDXL fallback)",
    )
    parser.add_argument(
        "--no-sdxl",
        action="store_true",
        help="Disable SDXL inpainting fallback (use only LaMa)",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=None,
        help="Reserved compatibility option for older unguided detection",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of images considered when choosing the preview (default: all)",
    )
    parser.add_argument(
        "--location-tolerance",
        type=float,
        default=None,
        metavar="RATIO",
        help="Maximum per-image position drift as an image ratio (default: 0.08)",
    )
    parser.add_argument(
        "--size-tolerance",
        type=float,
        default=None,
        metavar="RATIO",
        help="Maximum per-image blemish size drift (default: 0.12)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--skip-confirm",
        action="store_true",
        help="Skip interactive blemish confirmation (auto-accept)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    from src.config import Config
    from src.main import run_pipeline

    cfg = Config.from_cli_args(args)
    cfg.resolve()

    try:
        run_pipeline(cfg, skip_confirm=args.skip_confirm)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        if cfg.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())