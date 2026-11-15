# Phase 2 Preprocessing Report

## Summary

| Metric | Value |
|--------|-------|
| Source dataset | `E:\CV_Project\data\archive (1)` |
| Processed dataset | `E:\CV_Project\data\processed` |
| Processing time | 984.0 seconds |
| Total source images | 26,061 |
| Images processed OK | 26,061 |
| Images with errors | 0 |
| Images without source label | 3,257 |
| Label files written | 22,804 |
| Empty label files written | 2,046 |

## Annotation Filtering

| Class | Action | Count |
|-------|--------|-------|
| car (ID 2→0) | Kept & remapped | 65,439 |
| car (ID 0)   | Removed | 73,649 |
| dog (ID 1)   | Removed | 849 |
| other        | Removed | 0 |
| invalid format | Skipped | 0 |
| out of bounds  | Skipped | 0 |

## Per-Split Breakdown

| Split | Src Images | OK | Errors | No Label | Person Ann |
|-------|-----------|-----|--------|----------|------------|
| train | 23,287 | 23,287 | 0 | 3,257 | 55,073 |
| valid | 2,523 | 2,523 | 0 | 0 | 9,502 |
| test | 251 | 251 | 0 | 0 | 864 |

## Notes

- All processed images are **lossless single-channel (mode=L) PNG** files.
- Person class remapped: **source ID 2 → processed ID 0**.
- Images with no source label recorded in `missing_label_images.txt`.
- Original dataset files were **never modified**.