"""
preprocess_thermal_dataset.py
==============================
Phase 2 — Thermal Dataset Preprocessing Pipeline.

Purpose
-------
Converts the raw FLIR 2024 Roboflow YOLOv8 dataset into a clean, single-class
(person-only) single-channel (genuine grayscale) processed dataset suitable for
training the 1-channel YOLOv8 thermal pedestrian detector.

What this script does
---------------------
1.  Reads every thermal JPEG from archive (1)/train|valid|test/images/.
2.  Verifies the image is 640×480 uint8 with R=G=B (triplicated grayscale).
3.  Extracts channel 0 (the single unique thermal channel).
4.  Saves the result as a **lossless single-channel PNG** under data/processed/.
5.  Reads the corresponding YOLO .txt label file (if it exists).
6.  Filters annotations: keeps only class_id == 2 (person).
7.  Remaps class_id 2 → 0 (single class index for the new dataset).
8.  Writes the filtered label to data/processed/.
9.  Images with no source label → recorded in missing_label_images.txt only;
    NO synthetic annotation is created.
10. Source labels that exist but contain zero person annotations →
    an EMPTY .txt label file is written (valid YOLOv8 background sample).
11. Generates preprocessing_report.json and preprocessing_report.md.

Constraints
-----------
- Original files in data/archive (1)/ are NEVER modified, moved or deleted.
- If data/processed/ already has content the script aborts (safe re-run guard).
- Training is NOT started.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent           # e:\CV_Project
SOURCE_ROOT  = PROJECT_ROOT / "data" / "archive (1)"
DEST_ROOT    = PROJECT_ROOT / "data" / "processed"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
REPORTS_DIR  = PROJECT_ROOT / "scripts"  # reports written alongside scripts

SPLITS = ["train", "valid", "test"]

# Source class → keep/remap
SOURCE_PERSON_CLASS_ID = 2
DEST_PERSON_CLASS_ID   = 0

# Expected image properties
EXPECTED_W   = 640
EXPECTED_H   = 480
EXPECTED_DTYPE = "uint8"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%H:%M:%S")
    # Console — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    # File — DEBUG and above
    log_path = SCRIPTS_DIR / "preprocess.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Safety guard — prevent accidental overwrite
# ---------------------------------------------------------------------------
def check_dest_is_empty(logger: logging.Logger) -> None:
    """Abort if any processed images already exist."""
    for split in SPLITS:
        img_dir = DEST_ROOT / split / "images"
        if img_dir.exists() and any(img_dir.iterdir()):
            logger.error(
                f"Processed images already exist at {img_dir}. "
                "Delete data/processed/ contents or use --force to re-run."
            )
            sys.exit(1)
    logger.info("Safety check passed — processed output directories are empty.")


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------
def load_and_extract_channel(img_path: Path, logger: logging.Logger):
    """
    Loads a thermal JPEG, verifies it is triplicated grayscale (R=G=B),
    and returns a genuine 2-D uint8 NumPy array (H, W).

    Returns (array, warnings_list).
    """
    warnings = []
    try:
        img = Image.open(img_path)
    except Exception as e:
        raise IOError(f"Cannot open {img_path.name}: {e}")

    arr = np.array(img)

    # --- Dimension check ---
    if arr.ndim == 2:
        # Already single-channel — nothing to do
        if arr.shape != (EXPECTED_H, EXPECTED_W):
            warnings.append(
                f"Unexpected resolution {arr.shape[::-1]} in {img_path.name}; expected 640×480."
            )
        if arr.dtype != np.uint8:
            warnings.append(f"dtype is {arr.dtype}, expected uint8 in {img_path.name}.")
        return arr.astype(np.uint8), warnings

    if arr.ndim == 3:
        h, w, c = arr.shape
        if (h, w) != (EXPECTED_H, EXPECTED_W):
            warnings.append(
                f"Unexpected resolution {w}×{h} in {img_path.name}; expected 640×480."
            )
        if arr.dtype != np.uint8:
            warnings.append(f"dtype is {arr.dtype}, expected uint8 in {img_path.name}.")

        if c == 3:
            # Verify triplicated grayscale: R == G == B
            diff_rg = int(np.max(np.abs(arr[:, :, 0].astype(np.int32) - arr[:, :, 1].astype(np.int32))))
            diff_gb = int(np.max(np.abs(arr[:, :, 1].astype(np.int32) - arr[:, :, 2].astype(np.int32))))
            if diff_rg != 0 or diff_gb != 0:
                warnings.append(
                    f"[NOTICE] {img_path.name} channels differ (R-G max={diff_rg}, "
                    f"G-B max={diff_gb}). Using channel 0 as thermal."
                )
            # Extract single channel
            return arr[:, :, 0].astype(np.uint8), warnings

        if c == 1:
            return arr[:, :, 0].astype(np.uint8), warnings

        raise ValueError(f"Unexpected channel count {c} in {img_path.name}.")

    raise ValueError(f"Unexpected array ndim={arr.ndim} in {img_path.name}.")


def save_grayscale_png(arr_2d: np.ndarray, dest_path: Path) -> None:
    """Saves a 2D uint8 NumPy array as a lossless grayscale PNG."""
    img = Image.fromarray(arr_2d, mode="L")  # 'L' = 8-bit grayscale
    img.save(dest_path, format="PNG", optimize=False, compress_level=1)


# ---------------------------------------------------------------------------
# Label processing
# ---------------------------------------------------------------------------
def filter_and_remap_labels(src_label_path: Path):
    """
    Reads a YOLO .txt label file and returns person-only lines with
    class_id remapped from 2 → 0.

    Returns:
        person_lines  (list of str) — output lines to write
        stats         (dict)        — counts of kept / removed annotations
    """
    stats = {"total": 0, "person": 0, "car": 0, "dog": 0, "other": 0,
             "invalid": 0, "out_of_bounds": 0}
    person_lines = []

    try:
        raw = src_label_path.read_text(encoding="utf-8").strip()
    except Exception:
        stats["invalid"] += 1
        return person_lines, stats

    if not raw:
        return person_lines, stats

    for line in raw.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            stats["invalid"] += 1
            continue
        try:
            cid        = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            stats["invalid"] += 1
            continue

        stats["total"] += 1

        # Bounds check
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0
                and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            stats["out_of_bounds"] += 1
            # Still record class for stats, but skip writing
            if   cid == 0: stats["car"]  += 1
            elif cid == 1: stats["dog"]  += 1
            elif cid == 2: stats["person"] += 1
            else:          stats["other"] += 1
            continue

        if cid == SOURCE_PERSON_CLASS_ID:
            stats["person"] += 1
            person_lines.append(f"{DEST_PERSON_CLASS_ID} {xc} {yc} {w} {h}")
        elif cid == 0:
            stats["car"] += 1
        elif cid == 1:
            stats["dog"] += 1
        else:
            stats["other"] += 1

    return person_lines, stats


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------
def process_split(split: str, logger: logging.Logger) -> dict:
    """
    Processes a single dataset split (train/valid/test).
    Returns a per-split statistics dictionary.
    """
    src_img_dir = SOURCE_ROOT / split / "images"
    src_lbl_dir = SOURCE_ROOT / split / "labels"
    dst_img_dir = DEST_ROOT  / split / "images"
    dst_lbl_dir = DEST_ROOT  / split / "labels"

    # Collect all source images
    src_images = sorted(src_img_dir.glob("*.jpg"))
    logger.info(f"[{split}] Found {len(src_images):,} source images.")

    stats = {
        "split":                   split,
        "total_source_images":     len(src_images),
        "images_processed_ok":     0,
        "images_with_errors":      0,
        "images_with_warnings":    0,
        "images_no_source_label":  0,
        "labels_written":          0,
        "labels_empty_written":    0,
        "ann_total_source":        0,
        "ann_person_kept":         0,
        "ann_car_removed":         0,
        "ann_dog_removed":         0,
        "ann_other_removed":       0,
        "ann_invalid":             0,
        "ann_out_of_bounds":       0,
        "image_errors":            [],
        "image_warnings":          [],
        "missing_label_images":    [],
    }

    for idx, src_img in enumerate(src_images):
        stem = src_img.stem  # filename without extension
        dst_img_path = dst_img_dir / f"{stem}.png"
        dst_lbl_path = dst_lbl_dir / f"{stem}.txt"
        src_lbl_path = src_lbl_dir / f"{stem}.txt"

        # ---- Image ----
        try:
            arr, img_warnings = load_and_extract_channel(src_img, logger)
            save_grayscale_png(arr, dst_img_path)
            stats["images_processed_ok"] += 1
            if img_warnings:
                stats["images_with_warnings"] += 1
                stats["image_warnings"].extend(img_warnings)
                for w in img_warnings:
                    logger.debug(f"  IMG WARN: {w}")
        except Exception as e:
            stats["images_with_errors"] += 1
            msg = f"{src_img.name}: {e}"
            stats["image_errors"].append(msg)
            logger.warning(f"  IMG ERROR: {msg}")
            continue  # skip label processing for corrupt images

        # ---- Label ----
        if not src_lbl_path.exists():
            # No source label at all — record and move on; no label file written
            stats["images_no_source_label"] += 1
            stats["missing_label_images"].append(src_img.name)
            logger.debug(f"  NO LABEL: {src_img.name}")
            continue

        person_lines, lbl_stats = filter_and_remap_labels(src_lbl_path)

        stats["ann_total_source"]  += lbl_stats["total"]
        stats["ann_person_kept"]   += lbl_stats["person"]
        stats["ann_car_removed"]   += lbl_stats["car"]
        stats["ann_dog_removed"]   += lbl_stats["dog"]
        stats["ann_other_removed"] += lbl_stats["other"]
        stats["ann_invalid"]       += lbl_stats["invalid"]
        stats["ann_out_of_bounds"] += lbl_stats["out_of_bounds"]

        # Write label (empty string if no persons)
        dst_lbl_path.write_text(
            "\n".join(person_lines) + ("\n" if person_lines else ""),
            encoding="utf-8"
        )
        stats["labels_written"] += 1
        if not person_lines:
            stats["labels_empty_written"] += 1

        # Progress log every 2000 images
        if (idx + 1) % 2000 == 0:
            logger.info(
                f"  [{split}] Progress: {idx+1:,}/{len(src_images):,} images processed..."
            )

    logger.info(
        f"[{split}] Done — {stats['images_processed_ok']:,} ok | "
        f"{stats['images_with_errors']} errors | "
        f"{stats['images_no_source_label']:,} no-label | "
        f"{stats['ann_person_kept']:,} person annotations kept."
    )
    return stats


# ---------------------------------------------------------------------------
# data.yaml writer
# ---------------------------------------------------------------------------
def write_data_yaml() -> None:
    """Writes the new single-class data.yaml to data/processed/."""
    yaml_path = DEST_ROOT / "data.yaml"
    config = {
        "path":  str(DEST_ROOT).replace("\\", "/"),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    1,
        "names": ["person"],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def write_missing_labels_txt(all_stats: list) -> None:
    out_path = SCRIPTS_DIR / "missing_label_images.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Images with no corresponding source label file\n")
        f.write("# These images are treated as background-only negatives by YOLOv8.\n")
        f.write("# Do NOT invent annotations for these images.\n\n")
        for s in all_stats:
            split = s["split"]
            for fname in s["missing_label_images"]:
                f.write(f"{split}/images/{fname}\n")


def write_json_report(all_stats: list, elapsed: float) -> dict:
    total_src_imgs    = sum(s["total_source_images"]    for s in all_stats)
    total_ok          = sum(s["images_processed_ok"]    for s in all_stats)
    total_errors      = sum(s["images_with_errors"]     for s in all_stats)
    total_no_label    = sum(s["images_no_source_label"] for s in all_stats)
    total_lbl_written = sum(s["labels_written"]         for s in all_stats)
    total_lbl_empty   = sum(s["labels_empty_written"]   for s in all_stats)
    total_ann_src     = sum(s["ann_total_source"]       for s in all_stats)
    total_person      = sum(s["ann_person_kept"]        for s in all_stats)
    total_car         = sum(s["ann_car_removed"]        for s in all_stats)
    total_dog         = sum(s["ann_dog_removed"]        for s in all_stats)
    total_other       = sum(s["ann_other_removed"]      for s in all_stats)
    total_invalid     = sum(s["ann_invalid"]            for s in all_stats)
    total_oob         = sum(s["ann_out_of_bounds"]      for s in all_stats)

    report = {
        "phase":                   "2 — Thermal Dataset Preprocessing",
        "source_dataset":          str(SOURCE_ROOT),
        "processed_dataset":       str(DEST_ROOT),
        "elapsed_seconds":         round(elapsed, 1),
        "summary": {
            "total_source_images":     total_src_imgs,
            "images_processed_ok":     total_ok,
            "images_with_errors":      total_errors,
            "images_no_source_label":  total_no_label,
            "labels_written":          total_lbl_written,
            "labels_empty_written":    total_lbl_empty,
            "ann_total_source":        total_ann_src,
            "ann_person_kept":         total_person,
            "ann_car_removed":         total_car,
            "ann_dog_removed":         total_dog,
            "ann_other_removed":       total_other,
            "ann_invalid":             total_invalid,
            "ann_out_of_bounds":       total_oob,
        },
        "per_split": all_stats,
    }
    out = SCRIPTS_DIR / "preprocessing_report.json"
    with open(out, "w", encoding="utf-8") as f:
        # Truncate long error/warning lists in per_split for readability
        import copy
        safe = copy.deepcopy(report)
        for s in safe["per_split"]:
            s["image_errors"]   = s["image_errors"][:20]
            s["image_warnings"] = s["image_warnings"][:20]
            s["missing_label_images"] = s["missing_label_images"][:10]
        json.dump(safe, f, indent=4)
    return report


def write_md_report(report: dict) -> None:
    s = report["summary"]
    lines = [
        "# Phase 2 Preprocessing Report",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Source dataset | `{report['source_dataset']}` |",
        f"| Processed dataset | `{report['processed_dataset']}` |",
        f"| Processing time | {report['elapsed_seconds']} seconds |",
        f"| Total source images | {s['total_source_images']:,} |",
        f"| Images processed OK | {s['images_processed_ok']:,} |",
        f"| Images with errors | {s['images_with_errors']} |",
        f"| Images without source label | {s['images_no_source_label']:,} |",
        f"| Label files written | {s['labels_written']:,} |",
        f"| Empty label files written | {s['labels_empty_written']:,} |",
        "",
        "## Annotation Filtering",
        "",
        f"| Class | Action | Count |",
        f"|-------|--------|-------|",
        f"| car (ID 2→0) | Kept & remapped | {s['ann_person_kept']:,} |",
        f"| car (ID 0)   | Removed | {s['ann_car_removed']:,} |",
        f"| dog (ID 1)   | Removed | {s['ann_dog_removed']:,} |",
        f"| other        | Removed | {s['ann_other_removed']:,} |",
        f"| invalid format | Skipped | {s['ann_invalid']} |",
        f"| out of bounds  | Skipped | {s['ann_out_of_bounds']} |",
        "",
        "## Per-Split Breakdown",
        "",
        "| Split | Src Images | OK | Errors | No Label | Person Ann |",
        "|-------|-----------|-----|--------|----------|------------|",
    ]
    for sp in report["per_split"]:
        lines.append(
            f"| {sp['split']} | {sp['total_source_images']:,} | "
            f"{sp['images_processed_ok']:,} | {sp['images_with_errors']} | "
            f"{sp['images_no_source_label']:,} | {sp['ann_person_kept']:,} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- All processed images are **lossless single-channel (mode=L) PNG** files.",
        "- Person class remapped: **source ID 2 → processed ID 0**.",
        "- Images with no source label recorded in `missing_label_images.txt`.",
        "- Original dataset files were **never modified**.",
    ]
    out = SCRIPTS_DIR / "preprocessing_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = setup_logging()
    logger.info("=" * 65)
    logger.info("PHASE 2 — THERMAL DATASET PREPROCESSING")
    logger.info("=" * 65)

    # Guard: source must exist
    if not SOURCE_ROOT.exists():
        logger.error(f"Source dataset not found: {SOURCE_ROOT}")
        sys.exit(1)

    # Guard: do not overwrite existing processed data
    check_dest_is_empty(logger)

    t_start = time.time()

    all_stats = []
    for split in SPLITS:
        logger.info(f"\nProcessing split: [{split.upper()}]")
        split_stats = process_split(split, logger)
        all_stats.append(split_stats)

    elapsed = time.time() - t_start
    logger.info(f"\nAll splits complete in {elapsed:.1f}s.")

    # Write data.yaml
    write_data_yaml()
    logger.info(f"Written: {DEST_ROOT / 'data.yaml'}")

    # Write missing labels list
    write_missing_labels_txt(all_stats)
    logger.info(f"Written: {SCRIPTS_DIR / 'missing_label_images.txt'}")

    # Write reports
    report = write_json_report(all_stats, elapsed)
    write_md_report(report)
    logger.info(f"Written: {SCRIPTS_DIR / 'preprocessing_report.json'}")
    logger.info(f"Written: {SCRIPTS_DIR / 'preprocessing_report.md'}")

    # Final summary
    s = report["summary"]
    logger.info("\n" + "=" * 65)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 65)
    logger.info(f"  Images processed OK      : {s['images_processed_ok']:,}")
    logger.info(f"  Images with errors        : {s['images_with_errors']}")
    logger.info(f"  Images without label      : {s['images_no_source_label']:,}")
    logger.info(f"  Label files written       : {s['labels_written']:,}")
    logger.info(f"  Empty label files         : {s['labels_empty_written']:,}")
    logger.info(f"  Person annotations kept   : {s['ann_person_kept']:,}")
    logger.info(f"  Car annotations removed   : {s['ann_car_removed']:,}")
    logger.info(f"  Dog annotations removed   : {s['ann_dog_removed']:,}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()

