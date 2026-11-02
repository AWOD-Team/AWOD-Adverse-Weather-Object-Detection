#!/usr/bin/env bash
set -u
cd /data/fog_detect
source ~/anaconda3/etc/profile.d/conda.sh && conda activate torch2.4_cuda12.1
mkdir -p /data/awod_data/exp_v3_plan
exec python scripts/overnight_plan.py "$@"
