#!/bin/bash
# Ablation study on Promise12: budgets 3% (28 slices) and 4% (38 slices), seeds 1234 and 1233.
#   abl_alpha02 : full method, but --mse_param 0.2 (instead of 0.02)
#   abl_pgdr    : w/o PGDR (plain average-pooled features for clustering)
#   abl_gscu    : w/o GSCU (random sample per cluster instead of greedy uncertainty selection)
# The full-method reference runs (sugfw_{size}_seed{seed}) come from run_budgets.sh.
#
# Run from MA-SAM-un/:   bash command/Promise12/run_ablation.sh
# Needs the query lists from UC-SAM/make_ablation_lists.py. Resumable: finished runs are skipped.

export CUDA_VISIBLE_DEVICES=0

SIZES=(28 38)
SEEDS=(1234 1233)
DATA=../data/Promise12/Foreslices
SPL=$DATA/splits

# run <name> <csv> <seed> [extra args for train.py]
run() {
  name=$1; csv=$2; seed=$3; shift 3
  out=../output/Promise12/$name
  log=testing_log/Promise12/${name}__log.txt

  if [ ! -f "$csv" ]; then echo "!! missing $csv"; return; fi
  if [ -f "$log" ] && grep -q "Testing Finished" "$log"; then
    echo ">> $name already done, skipping"; return
  fi

  echo "=== $name: train ($(date +%H:%M:%S)) ==="
  python train.py -b 16 --n_gpu 1 \
    --root_path $DATA --csv_path "$csv" \
    -o "$out" --seed "$seed" --train_max_epochs 300 "$@" \
    || { echo "!! training failed for $name"; return; }

  echo "=== $name: test ($(date +%H:%M:%S)) ==="
  python inference.py --adapt_ckpt "$out/train_best_model.pth" \
    --root_path $DATA --stage test \
    --num_classes 2 --img_size 512 --seed "$seed" --cal_hd95 \
    || { echo "!! testing failed for $name"; return; }

  rm -f "$out"/epoch_*.pth
}

for size in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run abl_alpha02_${size}_seed${seed} $SPL/my_select_${size}_scale6_un10.csv        $seed --mse_param 0.2
    run abl_pgdr_${size}_seed${seed}    $SPL/my_select_${size}_wo_featureweighting.csv $seed
    run abl_gscu_${size}_seed${seed}    $SPL/my_select_${size}_wo_GS_seed${seed}.csv   $seed
  done
done

echo
echo "=== Summary: test Dice mean±std over the 30 test cases (HD95 is in voxels) ==="
for size in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for v in sugfw abl_alpha02 abl_pgdr abl_gscu; do
      f=testing_log/Promise12/${v}_${size}_seed${seed}__log.txt
      [ -f "$f" ] && echo "$v size=$size seed=$seed  $(grep '±' "$f" | tail -1 | sed 's/^\[[^]]*\] //')"
    done
  done
done
echo "All done."
