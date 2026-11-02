SUGFW+ Reproduction – ISIC 2018 Task 1

Overview

This project reproduces the SUGFW+ method from the original paper on the ISIC 2018 Task 1 skin lesion segmentation dataset.

The main goal was to reproduce the paper's segmentation results and investigate whether reducing the candidate pool used for uncertainty-based sample selection could save computation time without greatly affecting segmentation performance.

Two candidate pool sizes were tested:

Full pool: 2,594 training images.

Restricted pool: 315 training images, selected as a subset of the full training pool.

Method

The SUGFW+ pipeline was used for uncertainty-based sample selection and SAM-based segmentation.

Uncertainty maps were generated using 10 augmentations per image. The resulting uncertainty information was then used for sample selection with the SUGFW+ selection strategy.

For both candidate pools, four annotation budgets were tested:

Annotation budget

Selected images

0.5%

13

1%

26

2%

52

3%

78

The selected images were used to train the segmentation model. The same training configuration and test split were used for both candidate pool sizes.

Main training settings

Model: SAM ViT-B

Image size: 512 × 512

Batch size: 16

Epochs: 300

Learning rate: 0.0008

Rank: 32

Scale: 6

Random seed: 1233

Each configuration was trained once, using the same random seed.

Experiments

The experiments consisted of:

Generating uncertainty maps for all 2,594 training images.

Selecting samples from the full candidate pool at the four annotation budgets.

Selecting samples from the restricted pool of 315 images at the same budgets.

Training a segmentation model for each selected sample set.

Evaluating the trained models on the ISIC test split of 1,000 images.

Comparing Dice and HD95 with the results reported in the original paper.

Computational time

Uncertainty generation took approximately:

Candidate pool

Time

315 images

40 minutes

2,594 images

5–6 hours (estimated)

Using the smaller candidate pool reduced uncertainty-generation time by approximately 87–89%.

Results

The results showed that the full candidate pool did not consistently outperform the restricted pool.

The restricted pool achieved better mean Dice and HD95 at 0.5% and 1%.

The full pool achieved better mean Dice and HD95 at 2% and 3%.

The full-pool Dice results were closest to the paper at 2% and 3%.

Overall, reducing the candidate pool substantially lowered uncertainty-generation time, while producing reasonably comparable segmentation results in some annotation-budget settings.

Limitations

Only one random seed was used for each configuration, so the experiments do not show how much the results might vary across repeated runs.

The results should therefore be interpreted as a reproduction attempt and an initial comparison of the two candidate pool sizes, rather than a complete statistical evaluation.
