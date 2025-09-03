#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 1 (revised): detect & crop conjunctiva with Roboflow, fixing EXIF rotation
- Reads full-eye images, auto-rotates pixels per EXIF (so inference sees correct upright image),
  sends the rotated bytes to Roboflow, then crops from that same rotated image.
- No padding (per user). Picks highest-confidence 'conjunctiva', else highest overall.
- Outputs:
    cropped_images/<filename>          (upright cropped conjunctiva)
    outputs/step1_detections.csv       (one row per processed image with box & conf)
    outputs/step1_no_detections.csv    (images with no detection / error)
Docs:
  * Pillow exif_transpose: auto-apply EXIF Orientation & strip it afterwards.
  * Roboflow Hosted API (detect.roboflow.com) accepts base64 image data.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests
from PIL import Image, ImageOps

# ------------------------ Defaults ------------------------ #
DEFAULT_IMAGES_DIR = Path("images")
DEFAULT_OUT_DIR = Path("cropped_images")
DEFAULT_OUTPUTS_DIR = Path("outputs")
DEFAULT_MODEL_ID = "eye-conjunctiva-detector/2"
DEFAULT_CONFIDENCE_0_100 = 25          # ~0.25
DEFAULT_CLASS_NAME = "conjunctiva"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_SLEEP = 1.5               # seconds
# ---------------------------------------------------------- #


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    """Encode PIL image to JPEG bytes (RGB)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def roboflow_detect_b64(
    b64_str: str,
    model_id: str,
    api_key: str,
    confidence_0_100: int = DEFAULT_CONFIDENCE_0_100,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Call Roboflow Hosted API (Serverless) for object detection using base64 body.
    Endpoint: https://detect.roboflow.com/<model_id>?api_key=...&confidence=...
    """
    url = f"https://detect.roboflow.com/{model_id}"
    params = {
        "api_key": api_key,
        "confidence": str(confidence_0_100),  # API expects 0..100
        "format": "json",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(url, params=params, data=b64_str, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def select_best_box(preds: List[Dict[str, Any]], target_class: str) -> Optional[Dict[str, Any]]:
    if not preds:
        return None
    target = [p for p in preds if str(p.get("class", "")).lower() == target_class.lower()]
    if target:
        return max(target, key=lambda p: p.get("confidence", 0.0))
    return max(preds, key=lambda p: p.get("confidence", 0.0))


def crop_from_box(pil_img: Image.Image, box: Dict[str, Any]) -> Image.Image:
    """
    Roboflow boxes are center x,y with width,height in pixels for the input image.
    We already auto-rotated the pixels before sending, so coords align with pil_img here.
    """
    x = float(box["x"]); y = float(box["y"])
    w = float(box["width"]); h = float(box["height"])
    left = int(round(x - w / 2.0))
    top = int(round(y - h / 2.0))
    right = int(round(x + w / 2.0))
    bottom = int(round(y + h / 2.0))

    # clamp
    left = max(0, left); top = max(0, top)
    right = min(pil_img.width, right); bottom = min(pil_img.height, bottom)
    if right <= left or bottom <= top:
        return pil_img.copy()
    return pil_img.crop((left, top, right, bottom))


def process_one_image(
    path: Path,
    model_id: str,
    api_key: str,
    conf_0_100: int,
    class_name: str,
    max_retries: int,
    retry_sleep: float,
) -> Tuple[Optional[Image.Image], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns: (cropped PIL.Image or None, best_pred or None, meta dict)
    meta includes filename, orig_w, orig_h, rotated_w, rotated_h, and optional error
    """
    # 1) Load & auto-rotate pixels by EXIF (so inference & cropping see upright pixels)
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # returns new image if rotation needed
        img = img.convert("RGB")
        rotated_w, rotated_h = img.width, img.height
        # 2) Encode to JPEG bytes and base64 for Roboflow
        jpg_bytes = pil_to_jpeg_bytes(img, quality=90)
        b64 = base64.b64encode(jpg_bytes).decode("utf-8")
    except Exception as e:
        return None, None, {"filename": path.name, "error": f"load_or_rotate_failed: {e}"}

    # 3) Call Roboflow with retries
    retries = 0
    last_err = None
    while retries <= max_retries:
        try:
            data = roboflow_detect_b64(
                b64_str=b64,
                model_id=model_id,
                api_key=api_key,
                confidence_0_100=conf_0_100,
            )
            preds = data.get("predictions", [])
            best = select_best_box(preds, target_class=class_name)
            if best is None:
                return None, None, {
                    "filename": path.name,
                    "orig_w": rotated_w,
                    "orig_h": rotated_h,
                    "error": "no_detection",
                }
            crop = crop_from_box(img, best)
            meta = {
                "filename": path.name,
                "orig_w": rotated_w,
                "orig_h": rotated_h,
            }
            return crop, best, meta
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (429, 500, 502, 503, 504):
                last_err = e
                retries += 1
                time.sleep(retry_sleep * max(1, retries))
                continue
            raise
        except Exception as e:
            last_err = e
            retries += 1
            time.sleep(retry_sleep * max(1, retries))

    return None, None, {"filename": path.name, "error": f"roboflow_failed: {last_err}"}


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        header = ["filename", "class", "confidence", "x", "y", "width", "height", "orig_w", "orig_h", "note", "error"]
        out_path.write_text(",".join(header) + "\n", encoding="utf-8")
        return
    keys = set()
    for r in rows:
        keys.update(r.keys())
    ordered = ["filename","class","confidence","x","y","width","height","orig_w","orig_h","note","error"]
    for k in sorted(keys):
        if k not in ordered:
            ordered.append(k)
    lines = [",".join(ordered)]
    for r in rows:
        vals = []
        for k in ordered:
            v = r.get(k, "")
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                s = str(v).replace('"', '""')
                if "," in s or "\n" in s:
                    s = f'"{s}"'
                vals.append(s)
        lines.append(",".join(vals))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Step 1: crop conjunctiva via Roboflow (requests-only), fixing EXIF rotation.")
    parser.add_argument("--images-dir", type=str, default=str(DEFAULT_IMAGES_DIR), help="Folder with full-eye photos.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Folder to write cropped images.")
    parser.add_argument("--outputs-dir", type=str, default=str(DEFAULT_OUTPUTS_DIR), help="Folder to write CSV logs.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Roboflow model id, e.g. 'proj/2'")
    parser.add_argument("--api-key", type=str, default=os.getenv("ROBOFLOW_API_KEY", ""), help="Roboflow API key or set ROBOFLOW_API_KEY")
    parser.add_argument("--confidence", type=int, default=DEFAULT_CONFIDENCE_0_100, help="Threshold [0..100], default 25≈0.25")
    parser.add_argument("--class-name", type=str, default=DEFAULT_CLASS_NAME, help="Prefer this class when present")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-sleep", type=float, default=DEFAULT_RETRY_SLEEP)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    outputs_dir = Path(args.outputs_dir)
    model_id = args.model_id
    api_key = args.api_key.strip()
    conf_0_100 = int(args.confidence)
    class_name = args.class_name

    if not api_key:
        print("ERROR: Missing API key. Pass --api-key or set ROBOFLOW_API_KEY.", file=sys.stderr)
        sys.exit(2)
    if not images_dir.exists():
        print(f"ERROR: images folder not found: {images_dir}", file=sys.stderr)
        sys.exit(2)

    ensure_dirs(out_dir, outputs_dir)

    imgs = list_images(images_dir)
    if not imgs:
        print(f"No images found in {images_dir}")
        sys.exit(0)

    detections_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []

    total = len(imgs)
    print(f"Found {total} images. Processing (upright via EXIF) with model '{model_id}' ...")

    for i, img_path in enumerate(imgs, 1):
        try:
            crop, best, meta = process_one_image(
                path=img_path,
                model_id=model_id,
                api_key=api_key,
                conf_0_100=conf_0_100,
                class_name=class_name,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
            if crop is None or best is None:
                missing_rows.append({"filename": img_path.name, **(meta or {})})
                print(f"[{i}/{total}] NO DETECTION -> {img_path.name}")
                continue

            out_path = out_dir / img_path.name
            crop.save(out_path)  # saves upright crop
            row = {
                "filename": img_path.name,
                "class": best.get("class", ""),
                "confidence": float(best.get("confidence", 0.0)),
                "x": float(best.get("x", 0.0)),
                "y": float(best.get("y", 0.0)),
                "width": float(best.get("width", 0.0)),
                "height": float(best.get("height", 0.0)),
                "orig_w": meta.get("orig_w", ""),
                "orig_h": meta.get("orig_h", ""),
                "note": "cropped_upright",
                "error": "",
            }
            detections_rows.append(row)
            print(f"[{i}/{total}] ✓ {img_path.name}  conf={row['confidence']:.3f}  class={row['class']}")
        except Exception as e:
            missing_rows.append({"filename": img_path.name, "error": str(e)})
            print(f"[{i}/{total}] ERROR {img_path.name} -> {e}")
        time.sleep(0.15)  # be polite

    det_csv = outputs_dir / "step1_detections.csv"
    miss_csv = outputs_dir / "step1_no_detections.csv"
    save_csv(detections_rows, det_csv)
    save_csv(missing_rows, miss_csv)

    print("\nDone.")
    print(f"- Crops folder: {out_dir.resolve()}")
    print(f"- Detections CSV: {det_csv.resolve()}")
    print(f"- No-detections CSV: {miss_csv.resolve()}")
    print(f"Processed {len(detections_rows)} with crops; {len(missing_rows)} had no detection or error.")


if __name__ == "__main__":
    main()
