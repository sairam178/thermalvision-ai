# Phase 2 Validation Report

**Overall Status: ✅ ALL CHECKS PASSED**

## data.yaml

- ✅ All YAML checks passed.

## Per-Split Results

| Split | Images | Labels | Ann | Img Errors | Lbl Errors | Orphans | Status |
|-------|--------|--------|-----|-----------|-----------|---------|--------|
| train | 23,287 | 20,030 | 55,073 | 0 | 0 | 0 | ✅ |
| valid | 2,523 | 2,523 | 9,502 | 0 | 0 | 0 | ✅ |
| test | 251 | 251 | 864 | 0 | 0 | 0 | ✅ |

## Image Property Checks

- PIL mode = `L` (genuine 8-bit grayscale)
- `ndim` = 2 (no hidden channel axis)
- Shape = `(480, 640)`
- dtype = `uint8`

## Annotation Property Checks

- Only class ID 0 (person) present
- All coordinates normalised to [0, 1]
- Width and height are positive (> 0)
- Exactly 5 fields per line