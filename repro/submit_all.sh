#!/bin/bash
# Submit the 16 Table-1 cells (Qwen2.5-3B) as single-card jobs.
# The 4 Actor-Curator runs go to --qos=guaranteed-raul (1 card, never pre-empted, run one after another);
# the 12 baseline runs go to --qos=normal (up to 4 at once, pre-emptable, age-ordered queue).
set -euo pipefail
cd /shared/home/kseniia.kuvshinova/actor-curator
for cell in "countdown ac" "math ac" "arc ac" "zebra ac"; do
  set -- $cell; NGPU=1 repro/submit.sh "$1" "$2" --partition=gpu --qos=guaranteed-raul
done
for cell in "countdown uniform" "countdown sec" "countdown pcl" "math uniform" "math sec" "math pcl" \
            "arc uniform" "arc sec" "arc pcl" "zebra uniform" "zebra sec" "zebra pcl"; do
  set -- $cell; NGPU=1 repro/submit.sh "$1" "$2" --partition=gpu --qos=normal
done
squeue -u $USER -o "%.10i %.16q %.26j %.2t %.10M %R" | grep -E "ac-"
