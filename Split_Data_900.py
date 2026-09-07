import os
import shutil
import numpy as np
from pathlib import Path

SRC_DATA_DIR = Path(r"K:\FromDevexy2024-5-py\PyRepos\segmentation\DeepLock\Dataset_900\train_data\data")
SRC_GT_DIR = Path(r"K:\FromDevexy2024-5-py\PyRepos\segmentation\DeepLock\Dataset_900\train_data\ground-truth")

DST_ROOT = Path(r"K:\FromDevexy2024-5-py\PyRepos\segmentation\DeepLock\Dataset_900")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
SEED = 42

SPLITS = {"train": TRAIN_RATIO, "validation": VAL_RATIO, "test": TEST_RATIO}


def link_case(src_dir, dst_dir):
    """Hard-link a case folder without consuming extra disk space."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in os.listdir(str(src_dir)):
        src_f = src_dir / fname
        dst_f = dst_dir / fname
        if src_f.is_file():
            if not dst_f.exists():
                os.link(str(src_f), str(dst_f))


def main():
    case_ids = sorted([d.name for d in SRC_DATA_DIR.iterdir() if d.is_dir()])
    gt_ids = sorted([d.name for d in SRC_GT_DIR.iterdir() if d.is_dir()])
    assert case_ids == gt_ids, "data and ground-truth case lists do not match"
    n = len(case_ids)
    print(f"Cases available: {n}")

    rng = np.random.RandomState(SEED)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val
    assignments = {}
    for idx in indices[:n_train]:
        assignments[case_ids[idx]] = "train"
    for idx in indices[n_train:n_train + n_val]:
        assignments[case_ids[idx]] = "validation"
    for idx in indices[n_train + n_val:]:
        assignments[case_ids[idx]] = "test"
    print(f"Split sizes: train={n_train}, validation={n_val}, test={n_test}")

    for split in SPLITS:
        (DST_ROOT / split / "data").mkdir(parents=True, exist_ok=True)
        (DST_ROOT / split / "ground-truth").mkdir(parents=True, exist_ok=True)

    for case in case_ids:
        split = assignments[case]
        link_case(SRC_DATA_DIR / case, DST_ROOT / split / "data" / case)
        link_case(SRC_GT_DIR / case, DST_ROOT / split / "ground-truth" / case)

    for split in SPLITS:
        ids = [c for c, s in assignments.items() if s == split]
        with open(DST_ROOT / f"{split}.txt", "w") as f:
            for c in sorted(ids):
                f.write(f"{c}_lower\n{c}_upper\n")

    print("Done. Created Dataset_900 with train/validation/test.")
    for split in SPLITS:
        for sub in ("data", "ground-truth"):
            n_cases = sum(1 for _ in (DST_ROOT / split / sub).iterdir())
            n_objs = len(list((DST_ROOT / split / sub).rglob("*.obj")))
            print(f"  {split}/{sub}: {n_cases} cases, {n_objs} files")


if __name__ == "__main__":
    main()