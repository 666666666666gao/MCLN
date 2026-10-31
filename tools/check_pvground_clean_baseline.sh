set -euo pipefail
date -Is
screen -ls || true
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
ps -eo pid,ppid,pcpu,rss,etime,args | awk '/python/ && /train_pvground_clean_baseline.py/ && !/awk/ {print "pid=" $1, "parent=" $2, "cpu=" $3, "rss_kb=" $4, "elapsed=" $5}'
cd /root/autodl-tmp/pvground_clean_baseline_20261031
tail -n 8 train_clean.log
current=$(/root/autodl-tmp/PV-Ground/env/bin/python -c 'from pathlib import Path;print(max(Path("results").glob("*/*/*/config.json"),key=lambda p:int(p.parent.name)).parent)')
printf 'current_run=%s\n' "$current"
for name in first_update.json native_metrics.jsonl completed.json; do
  if [ -f "$current/$name" ]; then
    tail -30 "$current/$name"
  fi
done
