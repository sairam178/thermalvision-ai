import cv2
from pathlib import Path

def create_sample_video():
    test_dir = Path("data/processed/test/images")
    img_files = sorted(list(test_dir.glob("*.png")))[:120]
    
    if not img_files:
        print("[!] No test images found.")
        return

    out_dir = Path("uploads")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_thermal_pedestrians.mp4"

    first_frame = cv2.imread(str(img_files[0]))
    h, w = first_frame.shape[:2]

    # Use mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, 15.0, (w, h))

    for p in img_files:
        frame = cv2.imread(str(p))
        if frame is not None:
            out.write(frame)

    out.release()
    print(f"[+] Created sample thermal video with {len(img_files)} frames at: {out_path}")

if __name__ == "__main__":
    create_sample_video()

