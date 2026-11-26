#!/bin/bash
# Background watcher: appends job state transitions, errors and progress of all ac-* jobs to repro/logs/watch.log
cd /shared/home/kseniia.kuvshinova/actor-curator
declare -A last; declare -A pos
# start from the current end of every existing log (no replay of history)
for f in repro/logs/ac-*-*.out repro/logs/smoke-ac-*-*.out; do
  [ -f "$f" ] || continue; j=${f##*-}; j=${j%.out}; pos[$j]=$(wc -l < "$f")
  st=$(squeue -h -j "$j" -o "%T %R" 2>/dev/null); [ -z "$st" ] && st="ENDED($(sacct -n -X -j "$j" -o State 2>/dev/null | tr -d ' '))"; last[$j]=$st
done
echo "$(date +%m-%d\ %H:%M) watcher started (pid $$), tracking ${#pos[@]} logs"
while true; do
  for f in repro/logs/ac-*-*.out repro/logs/smoke-ac-*-*.out; do
    [ -f "$f" ] || continue
    j=${f##*-}; j=${j%.out}
    st=$(squeue -h -j "$j" -o "%T %R" 2>/dev/null); [ -z "$st" ] && st="ENDED($(sacct -n -X -j "$j" -o State 2>/dev/null | tr -d ' '))"
    if [ "$st" != "${last[$j]:-}" ]; then echo "$(date +%m-%d\ %H:%M) job $j ${f#repro/logs/}: $st"; last[$j]=$st; fi
    n=$(wc -l < "$f"); p=${pos[$j]:-0}
    [ "$n" -lt "$p" ] && { echo "$(date +%m-%d\ %H:%M) job $j log truncated (requeue?) - rescanning"; p=0; }
    if [ "$n" -gt "$p" ]; then
      tail -n +$((p+1)) "$f" | head -n $((n-p)) | grep -E "^Traceback|Error executing|RuntimeError|AssertionError|ValueError|out of memory|OutOfMemoryError|\bOOM\b|Killed|PREEMPT|val-core/|=== DONE|^=== [a-z_]+ +script=" | grep -v -E "FutureWarning|np.uintc" | while IFS= read -r line; do
        if [[ "$line" == *val-core/* ]]; then echo "$(grep -oE 'step:[0-9]+' <<<"$line" | head -1) $(grep -oE 'val-core/[^ ]+' <<<"$line" | tr '\n' ' ')"; else echo "${line:0:260}"; fi
      done | sed "s/^/$(date +%m-%d\ %H:%M) [$j] /"
      pos[$j]=$n
    fi
  done
  sleep 60
done
