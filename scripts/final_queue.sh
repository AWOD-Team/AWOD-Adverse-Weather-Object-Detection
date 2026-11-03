#!/usr/bin/env bash
# 等当前训练结束后自动接续：联合五折 -> 最终部署模型 -> 导出基准 -> 汇总
O=/data/awod_data/exp_v3_plan
while pgrep -f "scripts/overnight_plan.py" >/dev/null 2>&1; do sleep 60; done
echo "$(date '+%F %T') 队列开始 M/N/O/L" >> $O/queue.log
rm -f $O/done_L
bash /data/fog_detect/scripts/launch_stage.sh M N O L >> $O/pipeline.log 2>&1
echo "$(date '+%F %T') 队列结束" >> $O/queue.log
