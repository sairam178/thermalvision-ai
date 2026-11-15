import cv2
from pathlib import Path

def create_thermal_test_video():
    test_img_dir = Path("data/processed/test/images")
    test_lbl_dir = Path("data/processed/test/labels")
    
    # Select images that have pedestrian annotations
    positive_stems = []
    for lf in sorted(test_lbl_dir.glob("*.txt")):
        with open(lf) as f:
            lines = [l.strip() for l in f if l.strip()]
            if len(lines) > 0:
                positive_stems.append(lf.stem)

    print(f"Found {len(positive_stems)} test frames with pedestrians.")
    
    # Take 50-60 frames
    selected_stems = positive_stems[:60]
    img_paths = [test_img_dir / f"{s}.png" for s in selected_stems if (test_img_dir / f"{s}.png").exists()]

    if not img_paths:
        print("[!] No matching images found.")
        return

    out_dir = Path("uploads")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_thermal_patrol.mp4"

    # Read first frame for dimensions
    first = cv2.imread(str(img_paths[0]), cv2.IMREAD_GRAYSCALE)
    h, w = first.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Write at 10 fps so detections are clearly viewable
    writer = cv2.VideoWriter(str(out_path), fourcc, 10.0, (w, h), isColor=True)

    for p in img_paths:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        writer.write(bgr)

    writer.release()
    print(f"[+] Successfully generated thermal test video at: {out_path.resolve()}")
    print(f"    - Frames: {len(img_paths)}, Resolution: {w}x{h}, FPS: 10.0")

if __name__ == "__main__":
    create_thermal_test_video()

