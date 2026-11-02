import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

BASE_DIR = Path("../data/ISIC/Foreslices")
TRAIN_LIST = BASE_DIR / "splits" / "train_all.txt"
UNCERTAINTY_DIR = BASE_DIR / "uncertainty"
SAVE_PATH = UNCERTAINTY_DIR / "uncertainty_10_avg_full.pkl"

EXPECTED_COUNT = 2594
EXPECTED_SHAPE = (512, 512)


def main():
    if not TRAIN_LIST.is_file():
        raise FileNotFoundError(f"List not found: {TRAIN_LIST}")

    stems = [
        line.strip()
        for line in TRAIN_LIST.read_text().splitlines()
        if line.strip()
    ]

    if len(stems) != EXPECTED_COUNT or len(set(stems)) != EXPECTED_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COUNT} unique IDs; "
            f"found {len(stems)} entries and {len(set(stems))} unique IDs."
        )

    files = [UNCERTAINTY_DIR / f"{stem}_uncertainty_10.pkl" for stem in stems]
    missing = [p.name for p in files if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} uncertainty maps are missing. Examples: {missing[:10]}"
        )

    total = np.zeros(EXPECTED_SHAPE, dtype=np.float64)

    for path in tqdm(files, desc="Averaging 2594 uncertainty maps"):
        with path.open("rb") as f:
            uncertainty = np.asarray(pickle.load(f), dtype=np.float64)

        if uncertainty.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"{path.name}: expected {EXPECTED_SHAPE}, got {uncertainty.shape}"
            )

        if not np.isfinite(uncertainty).all():
            raise ValueError(f"Non-finite values in {path.name}")

        total += uncertainty

    average = total / len(files)

    with SAVE_PATH.open("wb") as f:
        pickle.dump(average, f)

    print(f"Saved: {SAVE_PATH}")
    print(f"Maps averaged: {len(files)}")
    print(f"Shape: {average.shape}")
    print(f"Value range: {average.min():.6f} to {average.max():.6f}")


if __name__ == "__main__":
    main()
