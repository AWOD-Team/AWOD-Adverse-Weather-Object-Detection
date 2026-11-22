"""Enhance one low-light image: python -m core.lowlight input.jpg output.png

Presets for different scenarios:
  --preset extreme    : For very dark scenes (high gain, contrast stretch)
  --preset moderate   : Default balanced enhancement
  --preset preserve   : Gentle enhancement preserving deep shadows
"""

import argparse
from pathlib import Path

from tools.dehaze_images import read_rgb, write_rgb

from . import LowLightRetinex


PRESETS = {
    "extreme": {
        "gain": 0.7,
        "shadow_rolloff": 12.0,
        "contrast_stretch": True,
        "denoise": "moderate",
        "help": "Very dark scenes with noise reduction (nighttime, tunnels)",
    },
    "moderate": {
        "gain": 0.5,
        "shadow_rolloff": 20.0,
        "contrast_stretch": True,
        "denoise": "mild",
        "help": "Balanced enhancement with gentle denoising (default)",
    },
    "preserve": {
        "gain": 0.35,
        "shadow_rolloff": 32.0,
        "contrast_stretch": False,
        "denoise": None,
        "help": "Gentle enhancement, preserves deep shadows",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="moderate",
        help="Enhancement strength preset",
    )
    parser.add_argument("--gain", type=float, help="Override preset gain")
    parser.add_argument("--shadow-rolloff", type=float, help="Override shadow rolloff")
    parser.add_argument(
        "--denoise",
        choices=["mild", "moderate", "strong"],
        help="Override denoising strength",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable denoising",
    )
    parser.add_argument(
        "--contrast-stretch",
        action="store_true",
        default=None,
        help="Enable contrast stretching",
    )
    parser.add_argument(
        "--no-contrast-stretch",
        action="store_false",
        dest="contrast_stretch",
        help="Disable contrast stretching",
    )

    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Choose a different output path to preserve the source image")

    # Build parameters from preset and overrides
    params = PRESETS[args.preset].copy()
    params.pop("help")
    if args.gain is not None:
        params["gain"] = args.gain
    if args.shadow_rolloff is not None:
        params["shadow_rolloff"] = args.shadow_rolloff
    if args.no_denoise:
        params["denoise"] = None
    elif args.denoise is not None:
        params["denoise"] = args.denoise
    if args.contrast_stretch is not None:
        params["contrast_stretch"] = args.contrast_stretch

    model = LowLightRetinex(**params)
    enhanced = model.enhance(read_rgb(args.input))
    write_rgb(args.output, enhanced)
    print(f"Enhanced with preset '{args.preset}': {args.output}")


if __name__ == "__main__":
    main()
