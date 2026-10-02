#!/usr/bin/env bash
set -u
export TZ=Europe/Berlin PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT=/home/josh/omp-workspace/drock-lmcache/lavd-qualification-20261001T231927Z
trap 'tmux wait-for -S lavd-remaining-finished-20261001T231927Z' EXIT
/usr/bin/python "$ROOT/campaign.py" --arms qad-tp4,spark-tp2,drafter-marlin,drafter-b12x,nccl-4,nccl-16,context-786k,scheduler-fixed,scheduler-old-stress,scheduler-fixed-stress,accuracy-off,dual-tp2 > "$ROOT/remaining-coordinator.log" 2>&1
rc=$?
printf '{"exit_code":%s}\n' "$rc" > "$ROOT/remaining-exit.json"
exit "$rc"
