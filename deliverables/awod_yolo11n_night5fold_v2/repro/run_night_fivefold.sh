#!/usr/bin/env bash
# 夜间数据集（night_2cls_clean_v2）分组五折 baseline：YOLO11n
# 每折流程：train.txt 训练 -> val.txt 早停/选 checkpoint -> test.txt 仅最终评测一次
# 超参按《数据集交付说明》第四节，五折完全一致。
#
# 用法: bash run_night_fivefold.sh [实验输出目录]
#   默认输出目录: /data/awod_data/exp_night5fold_v2
# 注意: data.yaml 内是绝对路径，换机器需重新生成 data.yaml 或调整 FOLDS。
set -u
source ~/anaconda3/etc/profile.d/conda.sh && conda activate torch2.4_cuda12.1

EXP=${1:-/data/awod_data/exp_night5fold_v2}
FOLDS=/data/awod_data/processed/night_2cls_clean_v2/folds5_grouped_v1
MODEL=/data/fog_detect/yolo11n.pt
CFG="imgsz=640 batch=16 seed=0 amp=True cache=True workers=8 device=0 exist_ok=True patience=30 lr0=0.001"

mkdir -p "$EXP/logs" "$EXP/runs" "$EXP/test"
cd /data/fog_detect || exit 1

for k in 1 2 3 4 5; do
  YAML=$FOLDS/fold$k/data.yaml
  echo "########## fold$k train  $(date '+%F %T') ##########"
  yolo detect train model=$MODEL data=$YAML epochs=100 project=$EXP/runs name=fold$k $CFG

  BEST=$EXP/runs/fold$k/weights/best.pt
  if [ ! -f "$BEST" ]; then echo "[fail] no $BEST"; continue; fi

  echo "########## fold$k test eval  $(date '+%F %T') ##########"
  python - "$BEST" "$YAML" "$EXP" "$k" <<'PY'
import json, sys
from ultralytics import YOLO
best, yaml, exp, k = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
m = YOLO(best)
r = m.val(data=yaml, split='test', imgsz=640, batch=16, device=0,
          project=f"{exp}/runs", name=f"fold{k}_test", exist_ok=True, plots=True)
d = {kk: float(vv) for kk, vv in r.results_dict.items()}
with open(f"{exp}/test/fold{k}.json", "w") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
print("TEST_METRICS fold%s " % k + json.dumps(d, ensure_ascii=False))
PY
done

echo "########## 汇总  $(date '+%F %T') ##########"
python - "$EXP" <<'PY'
import json, os, csv, sys, statistics as st
exp = sys.argv[1]
keys = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
rows = []
for k in range(1, 6):
    p = f"{exp}/test/fold{k}.json"
    if os.path.exists(p):
        rows.append((k, json.load(open(p))))
with open(f"{exp}/test_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fold"] + keys)
    for k, d in rows:
        w.writerow([k] + [f"{d.get(x, float('nan')):.6f}" for x in keys])
    for label, fn in (("mean", st.mean), ("std", st.pstdev)):
        vals = []
        for x in keys:
            xs = [d.get(x) for _, d in rows if d.get(x) is not None]
            vals.append(f"{fn(xs):.6f}" if xs else "")
        w.writerow([label] + vals)
agg = {}
for x in keys:
    xs = [d.get(x) for _, d in rows if d.get(x) is not None]
    if xs:
        agg[x] = {"mean": round(st.mean(xs), 6), "std": round(st.pstdev(xs), 6), "n": len(xs)}
with open(f"{exp}/test_summary.json", "w") as f:
    json.dump({"folds": {f"fold{k}": d for k, d in rows}, "aggregate": agg}, f, indent=2, ensure_ascii=False)
print(json.dumps(agg, indent=2, ensure_ascii=False))
PY

echo "ALL_DONE $(date '+%F %T')"
