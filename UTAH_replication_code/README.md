# UTAH Replication Code

This directory contains the code used to reproduce SUGFW+ experiments
on the UTAH dataset for left atrium segmentation.

## Contents
- MA-SAM-un/: training, inference, model and dataset implementation.
- UTAH_pre.py: UTAH dataset preprocessing.
- UTAH_csv.py: CSV generation for the UTAH dataset.
- save_spacing.py: voxel spacing utility.

## Experiments
The experiments used annotation budgets of 5, 8, 10, 13, 25 and 38 slices,
with the main runs trained for 300 epochs.

The dataset and trained model checkpoints are not included.

## Reproducibility
The commands, paths and configuration should be adapted to the local
dataset location and the environment used to run the experiments.
