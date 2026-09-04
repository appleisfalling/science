#!/bin/bash
# Sequential job queue for one GPU slot. Usage:
#   setsid bash run_queue.sh <queue_file> > /data/kimi_repro/logs/queue_<slot>.log 2>&1 < /dev/null &
# <queue_file>: one job per line:  <run_name> <dataset> <extra train_v2.py args...>
# Lines starting with # are skipped. A run whose directory already has best_model.pth + DONE marker is skipped
# (so a queue can be re-launched after interruption). Each run writes its own log to logs/<run_name>.log.
set -u
CODE=/data/kimi_repro/code/ukan_dg
PY=/data/kimi_repro/envs/ukan/bin/python
OUT=/data/kimi_repro/outputs/ukan_v2
LOGS=/data/kimi_repro/logs
Q=$1
mkdir -p "$OUT" "$LOGS"
cd "$CODE"
while IFS= read -r line || [ -n "$line" ]; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  set -- $line
  name=$1; ds=$2; shift 2
  if [ -f "$OUT/$name/DONE" ]; then echo "[queue] skip $name (DONE)"; continue; fi
  echo "[queue] $(date '+%F %T') START $name ($ds) $*"
  $PY train_v2.py --name "$name" --dataset "$ds" --data_dir inputs --output_dir "$OUT" "$@" > "$LOGS/$name.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && grep -q 'DONE best_val_iou' "$LOGS/$name.log"; then
    grep 'DONE best_val_iou' "$LOGS/$name.log" | tail -1 > "$OUT/$name/DONE"
    echo "[queue] $(date '+%F %T') END   $name rc=$rc $(cat "$OUT/$name/DONE")"
  else
    echo "[queue] $(date '+%F %T') FAIL  $name rc=$rc (see $LOGS/$name.log)"
  fi
done < "$Q"
echo "[queue] $(date '+%F %T') QUEUE_DONE $Q"
