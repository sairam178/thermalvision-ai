import torch
from pathlib import Path
import pprint

ckpt_path = Path("backend/models/best.pt")

print("=" * 70)
print(f"INSPECTING TRAINED CHECKPOINT: {ckpt_path.resolve()}")
print(f"File Size: {ckpt_path.stat().st_size:,} bytes ({ckpt_path.stat().st_size / (1024*1024):.2f} MB)")
print("=" * 70)

# 1. Inspect raw torch checkpoint dict with weights_only=False
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"\n[+] Checkpoint Type: {type(ckpt)}")

if isinstance(ckpt, dict):
    print(f"[+] Top-level Dictionary Keys ({len(ckpt.keys())}):")
    for k in ckpt.keys():
        v = ckpt[k]
        type_str = type(v).__name__
        if hasattr(v, "__len__") and not isinstance(v, (torch.Tensor, str)):
            print(f"    - '{k}': {type_str} (len={len(v)})")
        elif isinstance(v, torch.Tensor):
            print(f"    - '{k}': Tensor (shape={list(v.shape)}, dtype={v.dtype})")
        else:
            print(f"    - '{k}': {type_str} = {str(v)[:100]}")

    # Check model object vs state_dict
    model_obj = ckpt.get("model")
    print(f"\n[+] 'model' key type: {type(model_obj)}")
    is_serialized_module = isinstance(model_obj, torch.nn.Module)
    print(f"    -> Contains full serialized nn.Module architecture: {is_serialized_module}")

    # Inspect Ultralytics metadata
    print(f"\n[+] Checkpoint Metadata:")
    print(f"    - 'epoch': {ckpt.get('epoch')}")
    print(f"    - 'best_fitness': {ckpt.get('best_fitness')}")
    print(f"    - 'date': {ckpt.get('date')}")
    print(f"    - 'version': {ckpt.get('version')}")
    print(f"    - 'license': {ckpt.get('license')}")
    print(f"    - 'names' (Classes): {ckpt.get('names')}")

    # Inspect train_args
    train_args = ckpt.get("train_args")
    if train_args:
        print(f"\n[+] 'train_args' extracted:")
        if hasattr(train_args, "__dict__"):
            t_dict = train_args.__dict__
        elif isinstance(train_args, dict):
            t_dict = train_args
        else:
            t_dict = {}

        for key in ["model", "data", "epochs", "batch", "imgsz", "device", "optimizer", 
                    "hsv_h", "hsv_s", "hsv_v", "mosaic", "close_mosaic", "fliplr", "flipud", "channels"]:
            if key in t_dict:
                print(f"    - {key}: {t_dict[key]}")

    # 2. Check Input Channels & Weights
    if is_serialized_module:
        first_layer = None
        for name, m in model_obj.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                first_layer = m
                print(f"\n[+] First Convolutional Layer in Model: {name} -> {m}")
                print(f"    - in_channels: {m.in_channels}")
                print(f"    - out_channels: {m.out_channels}")
                print(f"    - kernel_size: {m.kernel_size}")
                print(f"    - stride: {m.stride}")
                print(f"    - weight tensor shape: {list(m.weight.shape)}")
                break

        # Check total parameter count
        total_params = sum(p.numel() for p in model_obj.parameters())
        trainable_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
        print(f"\n[+] Total Parameters: {total_params:,} (Trainable: {trainable_params:,})")

        # Check model.yaml
        if hasattr(model_obj, "yaml"):
            print(f"\n[+] model.yaml content / config:")
            yaml_content = getattr(model_obj, "yaml")
            if isinstance(yaml_content, dict):
                print(f"    - ch (input channels): {yaml_content.get('ch')}")
                print(f"    - nc (classes count) : {yaml_content.get('nc')}")
                print(f"    - names              : {yaml_content.get('names')}")
            else:
                print(f"    {str(yaml_content)[:300]}")

print("\n" + "=" * 70)
print("TESTING ULTRALYTICS HIGH-LEVEL API LOADING")
print("=" * 70)

from ultralytics import YOLO
import ultralytics
print(f"Installed Ultralytics version: {ultralytics.__version__}")
print(f"Installed PyTorch version: {torch.__version__}")

try:
    yolo_model = YOLO(str(ckpt_path))
    print("[PASS] YOLO('backend/models/best.pt') successfully loaded!")

    # Verify input layer via YOLO instance
    l0 = yolo_model.model.model[0].conv
    print(f"Verified Layer 0 in YOLO model: {l0} (in_channels={l0.in_channels})")
    assert l0.in_channels == 1, f"Expected in_channels == 1, got {l0.in_channels}"

    # Test forward pass with native resolution [1, 1, 480, 640]
    x_native = torch.randn(1, 1, 480, 640)
    with torch.no_grad():
        out_native = yolo_model.model(x_native)
    tensor_native = out_native[0] if isinstance(out_native, (list, tuple)) else out_native
    print(f"[PASS] Forward pass [1, 1, 480, 640] -> Shape: {list(tensor_native.shape)} (Expected [1, 5, 6300])")
    assert tensor_native.shape == (1, 5, 6300)

    # Test forward pass with square resolution [1, 1, 640, 640]
    x_square = torch.randn(1, 1, 640, 640)
    with torch.no_grad():
        out_square = yolo_model.model(x_square)
    tensor_square = out_square[0] if isinstance(out_square, (list, tuple)) else out_square
    print(f"[PASS] Forward pass [1, 1, 640, 640] -> Shape: {list(tensor_square.shape)} (Expected [1, 5, 8400])")
    assert tensor_square.shape == (1, 5, 8400)

    # Test predict on an actual processed test image
    test_imgs = list(Path("data/processed/test/images").glob("*.png"))
    if test_imgs:
        sample_img = str(test_imgs[0])
        print(f"\n[+] Testing predict() on actual FLIR test image: {sample_img}")
        res = yolo_model.predict(sample_img, conf=0.25, verbose=False)
        print(f"[PASS] Predict succeeded! Detections found: {len(res[0].boxes)}")
        if len(res[0].boxes) > 0:
            for b in res[0].boxes:
                print(f"    - Bounding Box: {b.xyxy.cpu().numpy()[0].round(1)}, Conf: {float(b.conf[0]):.3f}, Cls: {int(b.cls[0])}")

except Exception as e:
    print(f"[FAIL] Error during loading or inference: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)
