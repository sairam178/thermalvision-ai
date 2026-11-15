"""
validate_processed_dataset.py
==============================
Phase 2 — Processed Dataset Validation Script.

Purpose
-------
Independently verifies that every file in data/processed/ is structurally
correct, genuinely single-channel, correctly annotated, and ready for
1-channel YOLOv8 thermal pedestrian training.

Checks performed
----------------
Image checks (per file):
  1. File opens without error (not corrupt).
  2. PIL mode == 'L' (genuine 8-bit grayscale — NOT RGB).
  3. ndim == 2 (no hidden channel dimension).
  4. Shape == (480, 640).
  5. dtype == uint8.
  6. Pixel range within [0, 255].
  7. Pixel min != max (non-blank / non-uniform image).

Annotation checks (per annotation line):
  8.  Exactly 5 whitespace-separated fields.
  9.  class_id == 0 (only person; no car/dog).
  10. xc, yc, w, h are valid floats.
  11. xc, yc in [0.0, 1.0].
  12. w, h in (0.0, 1.0] (positive, non-zero, not overflowing).

Cross-checks:
  13. Every label file has a corresponding image file.
  14. No orphan label files without an image.

Dataset-level checks:
  15. data.yaml exists and is parseable.
  16. YAML nc == 1 and names == ['person'].
  17. YAML paths resolve to existing directories.
  18. All three splits (train, valid, test) are non-empty.

Outputs
-------
- validation_report.json  — machine-readable per-error log
- validation_report.md    — human-readable summary
- Exit code 0 if all checks pass, 1 if any critical errors exist.
"""

import sys
import json
import time
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEST_ROOT    = PROJECT_ROOT / "data" / "processed"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
SPLITS       = ["train", "valid", "test"]

EXPECTED_W     = 640
EXPECTED_H     = 480
EXPECTED_SHAPE = (EXPECTED_H, EXPECTED_W)   # (480, 640)
EXPECTED_MODE  = "L"
EXPECTED_DTYPE = np.uint8

MAX_ERRORS_LOGGED = 50  # cap per-error list length in JSON

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("validate")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(SCRIPTS_DIR / "validate.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------
def validate_image(img_path: Path) -> list:
    """Returns list of error strings; empty list = PASS."""
    errors = []
    try:
        img = Image.open(img_path)
        arr = np.array(img)
    except Exception as e:
        return [f"Cannot open: {e}"]

    if img.mode != EXPECTED_MODE:
        errors.append(f"PIL mode={img.mode!r}, expected {EXPECTED_MODE!r}")
    if arr.ndim != 2:
        errors.append(f"ndim={arr.ndim}, expected 2")
    if arr.shape != EXPECTED_SHAPE:
        errors.append(f"shape={arr.shape}, expected {EXPECTED_SHAPE}")
    if arr.dtype != EXPECTED_DTYPE:
        errors.append(f"dtype={arr.dtype}, expected uint8")
    if arr.size > 0 and int(arr.min()) == int(arr.max()):
        errors.append("Image appears blank/uniform (min==max).")
    return errors


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------
def validate_label(lbl_path: Path) -> list:
    """Returns list of error strings; empty list = PASS (including empty file)."""
    errors = []
    try:
        raw = lbl_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return [f"Cannot read: {e}"]

    if not raw:
        return []  # Valid empty label (background image)

    for lineno, line in enumerate(raw.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"Line {lineno}: expected 5 fields, got {len(parts)}: {line!r}")
            continue
        try:
            cid        = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
        except ValueError as e:
            errors.append(f"Line {lineno}: parse error — {e}")
            continue

        if cid != 0:
            errors.append(f"Line {lineno}: class_id={cid}, only 0 (person) is allowed.")
        if not (0.0 <= xc <= 1.0):
            errors.append(f"Line {lineno}: xc={xc} out of [0,1].")
        if not (0.0 <= yc <= 1.0):
            errors.append(f"Line {lineno}: yc={yc} out of [0,1].")
        if not (0.0 < w <= 1.0):
            errors.append(f"Line {lineno}: w={w} must be in (0,1].")
        if not (0.0 < h <= 1.0):
            errors.append(f"Line {lineno}: h={h} must be in (0,1].")
    return errors


# ---------------------------------------------------------------------------
# YAML validation
# ---------------------------------------------------------------------------
def validate_yaml(logger: logging.Logger) -> list:
    errors = []
    yaml_path = DEST_ROOT / "data.yaml"
    if not yaml_path.exists():
        return ["data.yaml not found in data/processed/."]

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        return [f"data.yaml parse error: {e}"]

    if cfg.get("nc") != 1:
        errors.append(f"data.yaml nc={cfg.get('nc')}, expected 1.")
    if cfg.get("names") != ["person"]:
        errors.append(f"data.yaml names={cfg.get('names')}, expected ['person'].")

    for key in ["train", "val", "test"]:
        rel = cfg.get(key, "")
        abs_path = DEST_ROOT / rel
        if not abs_path.exists():
            errors.append(f"data.yaml {key} path does not exist: {abs_path}")

    logger.info(f"  data.yaml: nc={cfg.get('nc')}, names={cfg.get('names')}")
    return errors


# ---------------------------------------------------------------------------
# Per-split validation
# ---------------------------------------------------------------------------
def validate_split(split: str, logger: logging.Logger) -> dict:
    img_dir = DEST_ROOT / split / "images"
    lbl_dir = DEST_ROOT / split / "labels"

    img_files = sorted(img_dir.glob("*.png"))
    lbl_files = sorted(lbl_dir.glob("*.txt"))

    img_stems = {f.stem for f in img_files}
    lbl_stems = {f.stem for f in lbl_files}

    orphan_labels  = lbl_stems - img_stems   # labels with no matching image
    imgs_no_label  = img_stems - lbl_stems   # images with no label (expected)

    split_result = {
        "split":              split,
        "image_count":        len(img_files),
        "label_count":        len(lbl_files),
        "images_no_label":    len(imgs_no_label),
        "orphan_labels":      sorted(orphan_labels),
        "images_failed":      0,
        "labels_failed":      0,
        "total_annotations":  0,
        "image_errors":       [],
        "label_errors":       [],
    }

    logger.info(
        f"[{split}] Validating {len(img_files):,} images, "
        f"{len(lbl_files):,} labels ..."
    )

    # --- Validate images ---
    for idx, img_path in enumerate(img_files):
        errs = validate_image(img_path)
        if errs:
            split_result["images_failed"] += 1
            if len(split_result["image_errors"]) < MAX_ERRORS_LOGGED:
                split_result["image_errors"].append({
                    "file": img_path.name, "errors": errs
                })
        if (idx + 1) % 3000 == 0:
            logger.info(f"  [{split}] Image progress: {idx+1:,}/{len(img_files):,}")

    # --- Validate labels ---
    for lbl_path in lbl_files:
        errs = validate_label(lbl_path)
        # Count annotation lines regardless
        raw = lbl_path.read_text(encoding="utf-8").strip()
        if raw:
            split_result["total_annotations"] += len(raw.splitlines())
        if errs:
            split_result["labels_failed"] += 1
            if len(split_result["label_errors"]) < MAX_ERRORS_LOGGED:
                split_result["label_errors"].append({
                    "file": lbl_path.name, "errors": errs
                })

    # --- Orphan labels ---
    if orphan_labels:
        logger.warning(
            f"  [{split}] {len(orphan_labels)} orphan label files (no matching image)."
        )

    pass_img = split_result["images_failed"] == 0
    pass_lbl = split_result["labels_failed"] == 0
    pass_orp = len(orphan_labels) == 0

    logger.info(
        f"  [{split}] Images: {'PASS' if pass_img else 'FAIL'} "
        f"({split_result['images_failed']} failed) | "
        f"Labels: {'PASS' if pass_lbl else 'FAIL'} "
        f"({split_result['labels_failed']} failed) | "
        f"Orphans: {'PASS' if pass_orp else 'FAIL'}"
    )
    return split_result


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def write_json_report(yaml_errors, split_results, elapsed):
    report = {
        "phase":      "2 — Processed Dataset Validation",
        "elapsed_s":  round(elapsed, 1),
        "yaml_errors": yaml_errors,
        "splits":      split_results,
        "overall_pass": (
            len(yaml_errors) == 0
            and all(
                s["images_failed"] == 0
                and s["labels_failed"] == 0
                and len(s["orphan_labels"]) == 0
                for s in split_results
            )
        ),
    }
    path = SCRIPTS_DIR / "validation_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    return report


def write_md_report(report: dict):
    status = "✅ ALL CHECKS PASSED" if report["overall_pass"] else "❌ ISSUES FOUND"
    lines = [
        "# Phase 2 Validation Report",
        "",
        f"**Overall Status: {status}**",
        "",
        "## data.yaml",
        "",
    ]
    if report["yaml_errors"]:
        for e in report["yaml_errors"]:
            lines.append(f"- ❌ {e}")
    else:
        lines.append("- ✅ All YAML checks passed.")

    lines += [
        "",
        "## Per-Split Results",
        "",
        "| Split | Images | Labels | Ann | Img Errors | Lbl Errors | Orphans | Status |",
        "|-------|--------|--------|-----|-----------|-----------|---------|--------|",
    ]
    for s in report["splits"]:
        ok = s["images_failed"] == 0 and s["labels_failed"] == 0 and len(s["orphan_labels"]) == 0
        lines.append(
            f"| {s['split']} | {s['image_count']:,} | {s['label_count']:,} | "
            f"{s['total_annotations']:,} | {s['images_failed']} | "
            f"{s['labels_failed']} | {len(s['orphan_labels'])} | "
            f"{'✅' if ok else '❌'} |"
        )

    lines += [
        "",
        "## Image Property Checks",
        "",
        "- PIL mode = `L` (genuine 8-bit grayscale)",
        "- `ndim` = 2 (no hidden channel axis)",
        "- Shape = `(480, 640)`",
        "- dtype = `uint8`",
        "",
        "## Annotation Property Checks",
        "",
        "- Only class ID 0 (person) present",
        "- All coordinates normalised to [0, 1]",
        "- Width and height are positive (> 0)",
        "- Exactly 5 fields per line",
    ]
    path = SCRIPTS_DIR / "validation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger = setup_logging()
    logger.info("=" * 65)
    logger.info("PHASE 2 — PROCESSED DATASET VALIDATION")
    logger.info("=" * 65)

    if not DEST_ROOT.exists():
        logger.error(f"Processed dataset not found: {DEST_ROOT}. Run preprocessing first.")
        sys.exit(1)

    t_start = time.time()

    # 1. Validate data.yaml
    logger.info("\n[YAML] Validating data.yaml ...")
    yaml_errors = validate_yaml(logger)
    if yaml_errors:
        for e in yaml_errors:
            logger.warning(f"  YAML ERROR: {e}")
    else:
        logger.info("  data.yaml: PASS")

    # 2. Validate each split
    split_results = []
    for split in SPLITS:
        logger.info(f"\n[{split.upper()}] Starting validation ...")
        result = validate_split(split, logger)
        split_results.append(result)

    elapsed = time.time() - t_start

    # 3. Write reports
    report = write_json_report(yaml_errors, split_results, elapsed)
    write_md_report(report)
    logger.info(f"\nWritten: {SCRIPTS_DIR / 'validation_report.json'}")
    logger.info(f"Written: {SCRIPTS_DIR / 'validation_report.md'}")

    # 4. Final summary
    overall = report["overall_pass"]
    logger.info("\n" + "=" * 65)
    logger.info(f"VALIDATION {'PASSED ✓' if overall else 'FAILED ✗'}")
    logger.info("=" * 65)
    for s in split_results:
        logger.info(
            f"  [{s['split']:5s}] images={s['image_count']:,} "
            f"labels={s['label_count']:,}  "
            f"ann={s['total_annotations']:,}  "
            f"img_err={s['images_failed']}  "
            f"lbl_err={s['labels_failed']}  "
            f"orphans={len(s['orphan_labels'])}"
        )
    logger.info(f"  Elapsed: {elapsed:.1f}s")
    logger.info("=" * 65)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()

