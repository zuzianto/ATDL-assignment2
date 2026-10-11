#!/bin/bash
# Train and test SUGFW+ on Promise12 for every annotation budget (3%..8%).
# Run from MA-SAM-un/:   bash command/Promise12/run_budgets.sh
#
# - Resumable: a budget/seed whose test already finished is skipped.
# - Needs the query lists my_select_<N>_scale6_un10.csv (made by UC-SAM/select_samples.py).
# - Deletes the intermediate epoch_*.pth files after testing to save disk,
#   keeps train_best_model.pth (so HD95 in mm can be recomputed later).

export CUDA_VISIBLE_DEVICES=0

# slices = int(960 * ratio):  3%  4%  5%  6%  7%  8%
SIZES=(28 38 48 57 67 76)
# three seeds, as in the reported results
SEEDS=(1234 1233 1235)

DATA=../data/Promise12/Foreslices
SPLITS=$DATA/splits

for seed in "${SEEDS[@]}"; do
  for size in "${SIZES[@]}"; do
    csv=$SPLITS/my_select_${size}_scale6_un10.csv
    name=sugfw_${size}_seed${seed}
    out=../output/Promise12/$name
    test_log=testing_log/Promise12/${name}__log.txt

    if [ ! -f "$csv" ]; then
      echo "!! missing $csv, skipping $name"
      continue
    fi
    if [ -f "$test_log" ] && grep -q "Testing Finished" "$test_log"; then
      echo ">> $name already done, skipping"
      continue
    fi

    echo "=== $name: train ($(date +%H:%M:%S)) ==="
    python train.py -b 16 --n_gpu 1 \
      --root_path $DATA \
      --csv_path "$csv" \
      -o "$out" --seed "$seed" --train_max_epochs 300 \
      || { echo "!! training failed for $name"; continue; }

    echo "=== $name: test ($(date +%H:%M:%S)) ==="
    python inference.py \
      --adapt_ckpt "$out/train_best_model.pth" \
      --root_path $DATA --stage test \
      --num_classes 2 --img_size 512 --seed "$seed" --cal_hd95 \
      || { echo "!! testing failed for $name"; continue; }

    rm -f "$out"/epoch_*.pth
  done
done

echo
echo "=== Summary (test set, mean±std over cases; HD95 in voxels) ==="
for seed in "${SEEDS[@]}"; do
  for size in "${SIZES[@]}"; do
    f=testing_log/Promise12/sugfw_${size}_seed${seed}__log.txt
    [ -f "$f" ] && echo "size=$size seed=$seed  $(grep '±' "$f" | tail -1 | sed 's/^\[[^]]*\] //')"
  done
done
echo "All done."
