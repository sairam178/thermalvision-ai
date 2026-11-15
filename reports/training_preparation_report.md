# Phase 4 Training Preparation Report: 1-Channel Thermal YOLOv8
## Thermal & Infrared Pedestrian Detection for Night Vision Surveillance
**Date:** 2026-11-02 | **Framework:** Ultralytics YOLOv8 8.4.198, PyTorch 2.15.0 | **Status:** ✅ **READY FOR COLAB TRAINING**

---

## 1. Executive Summary

Phase 4 prepares the project for training in **Google Colab** using GPU acceleration (Tesla T4 or better). All architectural components, dataloaders, configurations, and evaluation steps have been audited to ensure the custom 1-channel model trains natively within the Ultralytics engine without falling back to 3-channel RGB duplication.

- **Colab Training Notebook:** `notebooks/Thermal_YOLOv8_Colab_Training.ipynb`
- **Standalone Training Script:** `src/models/train_thermal.py`
- **Dataset Configuration:** `data/processed/data.yaml` (updated with `channels: 1`)
- **Model Checkpoint:** `weights/yolov8n_thermal_1ch.pt`
- **Architecture Config:** `configs/yolov8n_1ch.yaml`

---

## 2. Architecture & Dataloader Compatibility Audit

### 2.1 The Dataloader Single-Channel Guarantee
In default Ultralytics YOLOv8, `YOLODataset` reads images using OpenCV:
```python
im = imread(f, flags=self.cv2_flag)
```
Where `self.cv2_flag` is derived from:
```python
super().__init__(*args, channels=self.data.get("channels", 3), **kwargs)
self.cv2_flag = {1: cv2.IMREAD_GRAYSCALE, 3: cv2.IMREAD_COLOR}.get(channels, cv2.IMREAD_UNCHANGED)
```

By explicitly adding `channels: 1` into `data/processed/data.yaml`, `self.cv2_flag` evaluates to `0` (`cv2.IMREAD_GRAYSCALE`). The dataloader yields genuine single-channel tensors of shape `torch.Size([1, 640, 640])` into the batch.

### 2.2 First Convolution Compatibility
- **Layer 0 Conv:** `Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(2, 2))`
- **Batch Tensor Shape:** `(B, 1, 640, 640)`
- **Loss Forward Pass:** Verified. Target assignment and loss calculation (DFL, box loss, and classification loss) execute cleanly with 0 errors.

### 2.3 Single-Class Pedestrian Configuration
- `nc: 1`
- `names: ['person']`
- Verified that all ground-truth labels yield class index `0`.

---

## 3. Dataset Splits Audit

| Split | Images | Labels | Pedestrian Annotations | Background Negatives | Verified Channel Shape |
|---|---|---|---|---|---|
| **Train** | 23,287 | 20,030 | 55,073 | 5,104 (3,257 unlabelled + 1,847 empty) | `(1, 640, 640)` |
| **Valid** | 2,523 | 2,523 | 9,502 | 185 empty negatives | `(1, 640, 640)` |
| **Test** | 251 | 251 | 864 | 14 empty negatives | `(1, 640, 640)` |
| **Total** | **26,061** | **22,804** | **65,439** | **5,303** | `(1, 640, 640)` |

---

## 4. Hyperparameter Engineering for Thermal Infrared (LWIR)

Standard YOLOv8 augmentations are designed for RGB natural scenes. We tuned the training configuration specifically for single-channel thermal physics:

| Parameter | Default (RGB) | Thermal Setting | Justification |
|---|---|---|---|
| `hsv_h` | `0.015` | **`0.0`** | **Disabled.** Hue is mathematically non-existent in single-channel thermal data. |
| `hsv_s` | `0.7` | **`0.0`** | **Disabled.** Color saturation does not exist in LWIR sensors. |
| `hsv_v` | `0.4` | **`0.2`** | **Retained.** Brightness/intensity jitter simulates ambient thermal contrast and gain changes. |
| `fliplr` | `0.5` | **`0.5`** | **Retained.** Pedestrian posture is left-right symmetric. |
| `flipud` | `0.0` | **`0.0`** | **Disabled.** Pedestrians walk upright; vertical flipping degrades gravity semantics. |
| `mosaic` | `1.0` | **`1.0`** | **Retained.** Essential for detecting small, distant pedestrians. |
| `close_mosaic`| `10` | **`10`** | **Retained.** Disables mosaic in final 10 epochs for fine bounding box boundary convergence. |
| `optimizer` | `auto` | **`auto`** (AdamW) | Adaptive moment estimation with weight decay fine-tunes pretrained weights safely. |

---

## 5. Google Colab Training Notebook Structure

`notebooks/Thermal_YOLOv8_Colab_Training.ipynb` contains 23 executable and markdown cells structured into 11 logical steps:

1. **Step 1: GPU Audit:** Executes `!nvidia-smi` and checks CUDA capability and VRAM.
2. **Step 2: Dependency Installation:** Installs Ultralytics, OpenCV, Matplotlib, PyYAML.
3. **Step 3: Dataset Ingestion:** Unpacks `processed.zip` or mounts Google Drive.
4. **Step 4: Colab Dataset YAML:** Generates `data_colab.yaml` with absolute Colab paths and `channels: 1`.
5. **Step 5: Weight Transplantation:** Builds the 1-channel model and transplants weights from `yolov8n.pt`.
6. **Step 6: Pre-training Sanity Check:** Verifies Layer 0 Conv and output shape `(1, 5, 6300)`.
7. **Step 7: Training Session:** Executes `model.train()` with thermal hyperparameters for 50 epochs.
8. **Step 8: Metric Plots:** Visualizes `results.png`, `confusion_matrix.png`, and PR curves.
9. **Step 9: Test Split Evaluation:** Computes test mAP@0.50, mAP@0.50:0.95, Precision, Recall, and inference latency on 251 test frames.
10. **Step 10: Qualitative Thermal Detections:** Visualizes predictions with bounding boxes and confidence scores.
11. **Step 11: Export & Download:** Packages `best.pt` and training logs into a zip file for download.

---

## 6. How to Run Training on Google Colab

1. **Package Processed Dataset for Colab:**
   Create a zip archive of `data/processed`:
   ```powershell
   Compress-Archive -Path "e:\CV_Project\data\processed\*" -DestinationPath "e:\CV_Project\processed.zip"
   ```
2. **Open Colab:**
   Upload `notebooks/Thermal_YOLOv8_Colab_Training.ipynb` to Google Colab.
3. **Set Runtime to GPU:**
   Navigate to `Runtime` -> `Change runtime type` -> Select **T4 GPU**.
4. **Upload Data:**
   Upload `processed.zip` to `/content/` or place it in your Google Drive.
5. **Run All Cells:**
   Execute the notebook sequentially. Training 50 epochs on ~23k images takes approximately 45–60 minutes on a T4 GPU.
6. **Download Results:**
   The final cell automatically exports `thermal_yolov8_training_results.zip` containing `best.pt`.

