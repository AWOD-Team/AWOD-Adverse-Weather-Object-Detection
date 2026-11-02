#!/usr/bin/env bash
# Grouped five-fold baseline (YOLO11n) on fog_2cls_clean_v1.
# Each fold: train on train.txt, select checkpoint on val.txt, evaluate once on test.txt.
set -u
source ~/anaconda3/etc/profile.d/conda.sh && conda activate torch2.4_cuda12.1

EXP=/data/awod_data/exp_fog5fold_v2
FOLDS=/data/awod_data/processed/fog_2cls_clean_v1/folds5_grouped_v1
MODEL=/data/fog_detect/yolo11n.pt
CFG="imgsz=640 batch=16 seed=0 amp=True cache=True workers=8 device=0 exist_ok=True patience=30 lr0=0.001"

for k in 1 2 3 4 5; do
  YAML=$FOLDS/fold$k/data.yaml
  echo "########## fold$k train  $(date '+%F %T') ##########"
  yolo detect train model=$MODEL data=$YAML epochs=100 project=$EXP/runs name=fold$k $CFG

  BEST=$EXP/runs/fold$k/weights/best.pt
  if [ ! -f "$BEST" ]; then echo "[fail] no $BEST"; continue; fi
  echo "########## fold$k test eval  $(date '+%F %T') ##########"
  yolo detect val model=$BEST data=$YAML split=test project=$EXP/runs name=fold${k}_test exist_ok=True
done

echo "ALL_DONE $(date '+%F %T')"
