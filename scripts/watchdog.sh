#!/usr/bin/env bash
# 夜间流水线看门狗：每 5 分钟检查；进程不在了就重启（各阶段幂等，已完成会跳过）。
LOG=/data/awod_data/exp_v3_plan/watchdog.log
PLOG=/data/awod_data/exp_v3_plan/pipeline.log
mkdir -p /data/awod_data/exp_v3_plan
for i in $(seq 1 300); do
  if [ -f /data/awod_data/exp_v3_plan/FINISHED ]; then
    echo "$(date '+%F %T') 流水线已正常结束，看门狗退出" >> "$LOG"
    exit 0
  fi
  if ! pgrep -f "overnight_plan.py" >/dev/null 2>&1; then
    echo "$(date '+%F %T') 未检测到流水线进程，启动" >> "$LOG"
    setsid --fork bash /data/fog_detect/scripts/launch_stage.sh >> "$PLOG" 2>&1 < /dev/null
    sleep 60
  else
    age=$(( $(date +%s) - $(stat -c %Y "$PLOG" 2>/dev/null || echo "$(date +%s)") ))
    if [ "$age" -gt 2700 ]; then
      echo "$(date '+%F %T') 日志 $age 秒未更新，重启流水线" >> "$LOG"
      pkill -f "overnight_plan.py" 2>/dev/null
      sleep 10
    fi
  fi
  sleep 300
done
