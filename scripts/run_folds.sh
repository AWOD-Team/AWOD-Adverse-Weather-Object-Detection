#!/usr/bin/env bash
# 按折自动训练（两阶段微调：先冻结骨干，再全量小学习率微调）
# 用法: bash run_folds.sh <model.pt> <tag> <folds_dir> <stage1_epochs> <stage2_epochs> [额外 yolo 参数...]
set -uo pipefail
ENV=~/anaconda3/envs/torch2.4_cuda12.1
YOLO=$ENV/bin/yolo
MODEL=$1; TAG=$2; FOLDS=${3:-/data/fog_detect/folds}
E1=${4:-20}; E2=${5:-80}; shift 5 2>/dev/null || shift $#
RUNS=/data/fog_detect/runs
mkdir -p "$RUNS"
COMMON="imgsz=640 batch=16 patience=30 seed=0 deterministic=True cos_lr=True warmup_epochs=3 mosaic=1.0 mixup=0.15 degrees=10 scale=0.5 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 workers=8 device=0 exist_ok=True"

for k in 1 2 3 4 5; do
  YAML=$FOLDS/data_fold$k.yaml
  [ -f "$YAML" ] || { echo "[skip] 缺少 $YAML"; continue; }
  echo "########## $TAG fold$k : 阶段1（冻结骨干 $E1 epochs） ##########"
  $YOLO detect train model=$MODEL data=$YAML epochs=$E1 freeze=10 lr0=0.001 \
        project=$RUNS name=${TAG}_s1_fold$k $COMMON "$@"
  S1=$RUNS/${TAG}_s1_fold$k/weights/best.pt
  [ -f "$S1" ] || { echo "[fail] 阶段1 没产出 $S1"; continue; }
  echo "########## $TAG fold$k : 阶段2（全量微调 $E2 epochs） ##########"
  $YOLO detect train model=$S1 data=$YAML epochs=$E2 freeze=0 lr0=0.0008 \
        project=$RUNS name=${TAG}_s2_fold$k $COMMON "$@"
done
echo "全部折训练结束，运行结果聚合："
$ENV/bin/python /data/fog_detect/scripts/dataset_tools.py summarize --runs-dir $RUNS -o /data/fog_detect/runs/summary.csv