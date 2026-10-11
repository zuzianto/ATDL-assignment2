#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python train.py -b 16 --n_gpu 1 \
  --root_path ../data/Promise12/Foreslices \
  --csv_path ../data/Promise12/Foreslices/splits/my_select_28_scale6_un10.csv \
  -o ../output/Promise12/sugfw_28_seed1234 --seed 1234 --train_max_epochs 300

python inference.py \
  --adapt_ckpt ../output/Promise12/sugfw_28_seed1234/train_best_model.pth \
  --root_path ../data/Promise12/Foreslices --stage test \
  --num_classes 2 --img_size 512 --seed 1234 --cal_hd95

wait
echo "All tasks finished successfully."