"""Preprocess RGB images without YOLO; preserve names, subfolders and geometry.

Run from the repository root: python -m tools.dehaze_images --help
Output: images/, optional maps/, and run.json. Labels/splits are not generated.
"""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from core.dehaze import DarkChannelPrior, RetinexBDCP


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_rgb(path):
    # UNCHANGED avoids implicit EXIF rotation and preserves the source bit depth
    # so unsupported inputs fail explicitly instead of changing label geometry.
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected an 8-bit three-channel image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    options = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.suffix.lower() in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(path.suffix, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), options)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Image file or image directory")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--method", choices=("retinex_bdcp", "retinex_dcp_daylight", "dcp"),
                        default="retinex_bdcp")
    parser.add_argument("--omega", type=float, default=0.95)
    parser.add_argument("--t0", type=float, help="Minimum transmission (standard 0.1, daylight 0.45)")
    parser.add_argument("--local-weight", type=float,
                        help="Retinex illumination fraction (standard 0.5, daylight 0.15)")
    parser.add_argument("--bright-gamma", type=float, default=2.0, help="Brightness fusion exponent")
    parser.add_argument("--save-maps", action="store_true", help="Save Retinex-BDCP float32 maps as NPZ")
    args = parser.parse_args(argv)
    source, destination = args.input.resolve(), args.output.resolve()
    if not source.exists():
        parser.error(f"Input does not exist: {source}")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        parser.error("Output must be a new or empty directory; existing results are never overwritten")
    if source.is_dir() and (destination == source or source in destination.parents):
        parser.error("Output must be outside the input directory")
    if args.save_maps and args.method == "dcp":
        parser.error("--save-maps requires a Retinex method")
    root = source if source.is_dir() else source.parent
    paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS) \
        if source.is_dir() else [source]
    if not paths or any(p.suffix.lower() not in EXTENSIONS for p in paths):
        parser.error("No supported images found")
    if args.method != "dcp":
        mode = "daylight" if args.method == "retinex_dcp_daylight" else "standard"
        model = RetinexBDCP(omega=args.omega, t0=args.t0,
                            local_weight=args.local_weight, bright_gamma=args.bright_gamma,
                            mode=mode)
        parameters = model.parameters()
    else:
        model = DarkChannelPrior(omega=args.omega, t0=0.1 if args.t0 is None else args.t0)
        parameters = {"algorithm": "dcp", **vars(model)}
    report = {"status": "running", "input": str(source), "output": str(destination),
              "parameters": parameters, "numpy": np.__version__, "opencv": cv2.__version__,
              "image_encoding": "preserve extension; JPEG quality 95; no EXIF rotation",
              "image_count": len(paths), "processed": 0, "records": []}
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        for index, path in enumerate(paths, 1):
            relative = path.relative_to(root)
            image = read_rgb(path)
            tick = time.perf_counter()
            if args.method != "dcp":
                result, details = model.dehaze_with_details(image)
            else:
                result, details = model.dehaze(image), {}
            elapsed = time.perf_counter() - tick
            write_rgb(destination / "images" / relative, result)
            if args.save_maps:
                map_path = destination / "maps" / relative.with_suffix(relative.suffix + ".npz")
                map_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(map_path, **{k: v for k, v in details.items()
                                                if isinstance(v, np.ndarray)})
            report["records"].append({"image": relative.as_posix(), "shape": list(image.shape),
                                      "seconds": elapsed, "solver": details.get("solver")})
            report["processed"] = index
            if index == 1 or index % 20 == 0 or index == len(paths):
                print(f"Processed {index}/{len(paths)}", flush=True)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        (destination / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"Results: {destination}")


if __name__ == "__main__":
    main()
