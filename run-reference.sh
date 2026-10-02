#!/usr/bin/env bash
set -u
export TZ=Europe/Berlin PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT=/home/josh/omp-workspace/drock-lmcache/lavd-qualification-20261001T231927Z
trap 'tmux wait-for -S lavd-reference-finished-20261001T231927Z' EXIT
/usr/bin/python "$ROOT/campaign.py" --arms reference > "$ROOT/reference-coordinator.log" 2>&1
rc=$?
printf '{"exit_code":%s}\n' "$rc" > "$ROOT/reference-exit.json"
exit "$rc"
