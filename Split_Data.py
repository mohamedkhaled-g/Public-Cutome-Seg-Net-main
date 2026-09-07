import os
import shutil
import pandas as pd
from pathlib import Path

# ----------------------------
# Configuration
# ----------------------------
EXCEL_PATH = r"C:\Users\mohamed\Downloads\etst.xlsx"
SHEET_NAME = "Sheet1"

SRC_DATA_DIR = Path(r"C:\Users\mohamed\Downloads\train_data\data")
SRC_GT_DIR = Path(r"C:\Users\mohamed\Downloads\train_data\ground-truth")

DST_DATA_DIR = Path(r"K:\FromDevexy2024-5-py\PyRepos\segmentation\DeepLock\Dataset_600\test_data\data")
DST_GT_DIR = Path(r"K:\FromDevexy2024-5-py\PyRepos\segmentation\DeepLock\Dataset_600\test_data\ground-truth")

# Ensure destination directories exist
DST_DATA_DIR.mkdir(parents=True, exist_ok=True)
DST_GT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load filenames from Excel
# ----------------------------
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
filenames = df['Filename'].dropna().str.strip().tolist()

# Extract unique case IDs (e.g., "01328DDN" from "01328DDN_upper.obj")
case_ids = set()
for fname in filenames:
    if fname.endswith('.obj'):
        stem = fname.rsplit('_', 1)[0]  # Splits on last '_' → ['01328DDN', 'upper.obj'] → take first
        case_ids.add(stem)

print(f"Found {len(case_ids)} unique cases to move.")

# ----------------------------
# Move data and ground truth folders
# ----------------------------
moved_cases = 0
for case in case_ids:
    # Define source and destination paths for data
    src_data_folder = SRC_DATA_DIR / case
    dst_data_folder = DST_DATA_DIR / case

    # Define source and destination paths for ground truth
    src_gt_folder = SRC_GT_DIR / case
    dst_gt_folder = DST_GT_DIR / case

    try:
        # Move data folder if it exists
        if src_data_folder.exists() and src_data_folder.is_dir():
            shutil.move(str(src_data_folder), str(dst_data_folder))
            print(f"✅ Moved data folder: {case}")
        else:
            print(f"⚠️  Data folder not found: {src_data_folder}")

        # Move ground-truth folder if it exists
        if src_gt_folder.exists() and src_gt_folder.is_dir():
            shutil.move(str(src_gt_folder), str(dst_gt_folder))
            print(f"✅ Moved GT folder: {case}")
        else:
            print(f"⚠️  GT folder not found: {src_gt_folder}")

        moved_cases += 1

    except Exception as e:
        print(f"❌ Error moving case {case}: {e}")

print(f"\n✅ Completed! Moved {moved_cases} out of {len(case_ids)} cases.")