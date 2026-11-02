
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


BASE_DIR = Path("../data/ISIC/Foreslices")
PILOT_LIST = BASE_DIR / "splits" / "pilot_315.txt"
UNCERTAINTY_DIR = BASE_DIR / "uncertainty"
SAVE_PATH = UNCERTAINTY_DIR / "uncertainty_10_avg.pkl"

EXPECTED_SHAPE = (512, 512)


def main():
    if not PILOT_LIST.is_file():
        raise FileNotFoundError(f"Pilot list not found: {PILOT_LIST}")

    stems = [
        line.strip()
        for line in PILOT_LIST.read_text().splitlines()
        if line.strip()
    ]

    if len(stems) != 315 or len(set(stems)) != 315:
        raise ValueError(
            f"Expected 315 unique image IDs; found {len(stems)} entries "
            f"and {len(set(stems))} unique IDs."
        )

    files = [
        UNCERTAINTY_DIR / f"{stem}_uncertainty_10.pkl"
        for stem in stems
    ]

    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} uncertainty maps are missing. "
            f"Example: {missing[:5]}"
        )

    total = np.zeros(EXPECTED_SHAPE, dtype=np.float64)

    for path in tqdm(files, desc="Averaging 315 uncertainty maps"):
        with path.open("rb") as f:
            uncertainty = np.asarray(pickle.load(f), dtype=np.float64)

        if uncertainty.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"{path.name}: expected {EXPECTED_SHAPE}, "
                f"got {uncertainty.shape}"
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
