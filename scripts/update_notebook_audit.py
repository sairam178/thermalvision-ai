import json
from pathlib import Path

nb_path = Path("notebooks/Thermal_YOLOv8_Colab_Training.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Step 4 Markdown (Cell 7)
nb["cells"][7]["source"] = [
    "### Step 4: Verify Dataset Split Counts & Background Negatives\n",
    "Audit all images, labels, and background negatives across splits.\n",
    "\n",
    "#### Dataset Split & Background Negatives Table\n",
    "| Split | Total Images | Positive Frames (With Pedestrians) | Background Negatives (Zero Annotations) | Pedestrian Annotations |\n",
    "|:---|:---:|:---:|:---:|:---:|\n",
    "| **Train** | 23,287 | 18,183 | **5,104** (3,257 missing labels + 1,847 empty labels) | 57,801 |\n",
    "| **Validation** | 2,523 | 2,338 | **185** (0 missing labels + 185 empty labels) | 6,861 |\n",
    "| **Test** | 251 | 237 | **14** (0 missing labels + 14 empty labels) | 777 |\n",
    "| **TOTAL** | **26,061** | **20,758** | **5,303** | **65,439** |\n",
    "\n",
    "> ### Critical Academic Note & Limitation Analysis: 3,257 Unlabelled Training Images\n",
    "> In the FLIR 2024 source training split, **3,257 images** have no corresponding annotation `.txt` files.\n",
    "> \n",
    "> **Potential Limitation & Label Noise:**\n",
    "> - In real-world thermal surveillance datasets, unannotated frames often correspond to empty road sequences where no objects were observed. However, there is an inherent risk that some frames may contain faint, occluded, or distant pedestrians that human annotators missed.\n",
    "> - **Impact on Training:** In YOLOv8, an unlabelled image is treated as a 100% negative background frame. If an unannotated pedestrian is present, the model's correct detection will be penalized by the loss function as a false positive, potentially reducing sensitivity to small/distant pedestrians.\n",
    "> - **Engineering Tradeoff:** Retaining these 3,257 frames as hard background negatives teaches the model to suppress false alarms on heated vehicle engines, asphalt reflections, and warm building facades (a critical requirement for thermal surveillance). Retaining them without fabricating synthetic annotations preserves dataset integrity."
]

# Update Step 4 Code (Cell 8)
nb["cells"][8]["source"] = [
    "import os\n",
    "import glob\n",
    "from pathlib import Path\n",
    "\n",
    "data_root = PROJECT_DIR / \"data\" / \"processed\"\n",
    "splits = [\"train\", \"valid\", \"test\"]\n",
    "\n",
    "print(\"=\" * 75)\n",
    "print(\"DATASET INTEGRITY & BACKGROUND NEGATIVES AUDIT\")\n",
    "print(\"=\" * 75)\n",
    "total_imgs = 0\n",
    "total_pos_imgs = 0\n",
    "total_neg_imgs = 0\n",
    "total_lbl_files = 0\n",
    "total_persons = 0\n",
    "\n",
    "expected_negatives = {\"train\": 5104, \"valid\": 185, \"test\": 14}\n",
    "\n",
    "for sp in splits:\n",
    "    img_files = sorted(glob.glob(str(data_root / sp / \"images\" / \"*.png\")))\n",
    "    lbl_files = sorted(glob.glob(str(data_root / sp / \"labels\" / \"*.txt\")))\n",
    "    lbl_set = {Path(f).stem: f for f in lbl_files}\n",
    "    \n",
    "    split_pos_imgs = 0\n",
    "    split_neg_imgs = 0\n",
    "    split_persons = 0\n",
    "    \n",
    "    for img_path in img_files:\n",
    "        stem = Path(img_path).stem\n",
    "        if stem not in lbl_set:\n",
    "            # Missing label file = negative background frame\n",
    "            split_neg_imgs += 1\n",
    "        else:\n",
    "            with open(lbl_set[stem], \"r\") as f:\n",
    "                lines = [l.strip() for l in f if l.strip()]\n",
    "            if len(lines) == 0:\n",
    "                # Empty label file = negative background frame\n",
    "                split_neg_imgs += 1\n",
    "            else:\n",
    "                split_pos_imgs += 1\n",
    "                for line in lines:\n",
    "                    assert line.startswith(\"0 \"), f\"Invalid class ID in {lbl_set[stem]}! Must be 0.\"\n",
    "                    split_persons += 1\n",
    "    \n",
    "    total_imgs += len(img_files)\n",
    "    total_lbl_files += len(lbl_files)\n",
    "    total_pos_imgs += split_pos_imgs\n",
    "    total_neg_imgs += split_neg_imgs\n",
    "    total_persons += split_persons\n",
    "    \n",
    "    print(f\"[{sp.upper():5s}] Images: {len(img_files):>6,} | Positive: {split_pos_imgs:>6,} | Negatives: {split_neg_imgs:>5,} | Pedestrians: {split_persons:>6,}\")\n",
    "    if len(img_files) > 0:\n",
    "        assert split_neg_imgs == expected_negatives[sp], f\"{sp} negative count mismatch: {split_neg_imgs} != {expected_negatives[sp]}\"\n",
    "\n",
    "print(\"-\" * 75)\n",
    "print(f\"TOTAL  Images: {total_imgs:>6,} | Positive: {total_pos_imgs:>6,} | Negatives: {total_neg_imgs:>5,} | Pedestrians: {total_persons:>6,}\")\n",
    "print(\"=\" * 75)\n",
    "if total_imgs > 0:\n",
    "    assert total_imgs == 26061, f\"Expected 26,061 total images, got {total_imgs}\"\n",
    "    assert total_neg_imgs == 5303, f\"Expected 5,303 total background negatives, got {total_neg_imgs}\"\n",
    "    assert total_pos_imgs == 20758, f\"Expected 20,758 positive images, got {total_pos_imgs}\"\n",
    "    assert total_persons == 65439, f\"Expected 65,439 pedestrian annotations, got {total_persons}\"\n",
    "    print(\"[PASS] All split image, background negative, and annotation counts match audit targets exactly.\")\n"
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully updated notebooks/Thermal_YOLOv8_Colab_Training.ipynb")

