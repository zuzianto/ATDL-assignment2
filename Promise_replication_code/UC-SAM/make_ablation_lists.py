"""Create the query lists for the Promise12 ablation study (3% and 4% budgets).

Writes into ../data/Promise12/Foreslices/splits/:
  my_select_{n}_wo_featureweighting.csv     w/o PGDR  (plain average-pooled features; deterministic)
  my_select_{n}_wo_GS_seed{seed}.csv        w/o GSCU  (random pick inside each cluster; one list per seed)

The full-method lists (my_select_{n}_scale6_un10.csv) are reused for the alpha_max ablation.
Run from UC-SAM/:   python make_ablation_lists.py
"""
import os
import shutil
import sys

import numpy as np

from select_samples import Sample_Selector

SPLITS = "../data/Promise12/Foreslices/splits"
BUDGETS = [(0.03, 28), (0.04, 38)]  # (ratio, number of slices out of 960)
SEEDS = [1234, 1233]


def main(splits: str = SPLITS):
    train_csv = os.path.join(splits, "train.csv")
    for ratio, n in BUDGETS:
        selector = Sample_Selector(train_csv, select_ratio=ratio)
        assert selector.select_num == n, (selector.select_num, n)

        # w/o PGDR: the clustering + greedy selection stay, only the feature pooling changes
        selector.sugfw_wo_featureweighting_select()

        # w/o GSCU: random sample per cluster, so seed numpy and keep one list per seed
        for seed in SEEDS:
            np.random.seed(seed)
            selector.sugfw_wo_GS_select(soft_scale=6)
            shutil.copy(
                os.path.join(splits, f"my_select_{n}_wo_GS.csv"),
                os.path.join(splits, f"my_select_{n}_wo_GS_seed{seed}.csv"),
            )

    print("\nCreated:")
    for name in sorted(os.listdir(splits)):
        if "wo_featureweighting" in name or "wo_GS_seed" in name:
            print("  ", name)


if __name__ == "__main__":
    main(*sys.argv[1:])
