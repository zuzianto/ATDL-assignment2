from pathlib import Path
import pandas as pd

from select_samples import Sample_Selector

BASE = Path("../data/ISIC/Foreslices")
PILOT_CSV = BASE / "splits" / "pilot_315.csv"

# These budgets are calculated from the full training set of 2,594 images.
BUDGETS = {
    0.5: 13,
    1.0: 26,
    2.0: 52,
    3.0: 78,
}

df = pd.read_csv(PILOT_CSV)
print(f"Pilot candidates: {len(df)}")

for budget_percent, count in BUDGETS.items():
    print(f"\nBudget {budget_percent}%: selecting {count} images")

    # The original selector calculates its cluster count from the CSV size.
    # Build a temporary CSV whose ratio produces the desired count.
    selector = Sample_Selector(
        csv_path=str(PILOT_CSV),
        select_ratio=count / len(df),
    )

    selector.sugfw_select(soft_scale=6, un=10)

    generated = Path(
        f"../data/ISIC/Foreslices/splits/"
        f"my_select_{count}_scale6_un10.csv"
    )

    if not generated.exists():
        raise FileNotFoundError(
            f"Expected selection output was not created: {generated}"
        )

    selected = pd.read_csv(generated)
    if len(selected) != count:
        raise RuntimeError(
            f"Expected {count} selected images, got {len(selected)}"
        )

    # Rename to clearly identify this as a pilot selection.
    output = BASE / "splits" / f"pilot_selected_{budget_percent:g}pct.csv"
    selected.to_csv(output, index=False)

    print(f"Saved {len(selected)} images to {output}")

print("\nAll pilot selections completed.")
