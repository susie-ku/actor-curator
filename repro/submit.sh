#!/bin/bash
# usage: repro/submit.sh BENCH METHOD [SMOKE=1] [extra sbatch args...]
#   e.g. repro/submit.sh countdown ac
#        SMOKE=1 repro/submit.sh countdown ac --partition=debug --time=01:00:00
set -euo pipefail
cd /shared/home/kseniia.kuvshinova/actor-curator
BENCH=$1; METHOD=$2; shift 2
SMOKE=${SMOKE:-0}; SEED=${SEED:-1}; EXTRA=${EXTRA:-}; NGPU=${NGPU:-2}
if [ "$NGPU" = 1 ]; then RES=(--gres=shard:10 --cpus-per-task=16 --mem=128G); else RES=(--gres=gpu:2 --cpus-per-task=32 --mem=240G); fi
NAME="ac-${BENCH}-${METHOD}"; [ "$SMOKE" = 1 ] && NAME="smoke-${NAME}"
sbatch --job-name="$NAME" "${RES[@]}" --export=ALL,BENCH="$BENCH",METHOD="$METHOD",SMOKE="$SMOKE",SEED="$SEED",EXTRA="$EXTRA",NGPU="$NGPU" "$@" repro/run.sh
