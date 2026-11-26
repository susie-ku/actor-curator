#!/bin/bash
#SBATCH --job-name=ac-greedy-eval
#SBATCH --gres=shard:10
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=raul
#SBATCH --qos=guaranteed-raul
#SBATCH --partition=gpu
#SBATCH --nodelist=mbz-titan-1
#SBATCH --time=05:00:00
#SBATCH --output=repro/logs/%x-%j.out
#SBATCH --error=repro/logs/%x-%j.out
#
# Greedy (temperature 0) validation on math_test.parquet (MATH500 + AMC + AIME24) with the repo's own
# pipeline (prompt, verifier, vLLM) in trainer.val_only mode, for the untrained Qwen2.5-3B and the step-100
# checkpoints of the four MATH runs (all saved on titan-1's /scratch). Compares with the paper's pi_ref 61.8 %.
set -uo pipefail
REPO=/shared/home/kseniia.kuvshinova/actor-curator; cd "$REPO"
source /shared/home/kseniia.kuvshinova/miniconda3/etc/profile.d/conda.sh; conda activate verl
export HF_HOME=/shared/models/huggingface HF_HUB_OFFLINE=1
CK=/scratch/$USER/actor-curator-ckpt
echo "=== greedy eval on $(hostname) $(date) ==="; ls $CK

declare -A MODEL=( [base]=Qwen/Qwen2.5-3B )
for m in ac sec pcl uniform; do
  src=$CK/math_${m}_qwen2.5-3b_s1/global_step_100/actor; dst=$CK/greedy_eval/math_${m}_step100_hf
  if [ -d "$dst" ] && [ -f "$dst/config.json" ]; then MODEL[$m]=$dst; continue; fi
  if [ ! -d "$src" ]; then echo "!! no checkpoint for $m at $src"; continue; fi
  echo "--- merging $src -> $dst"
  python -m verl.model_merger merge --backend fsdp --local_dir "$src" --target_dir "$dst" && MODEL[$m]=$dst || echo "!! merge failed for $m"
done

for m in base ac sec pcl uniform; do
  [ -n "${MODEL[$m]:-}" ] || continue
  echo "=== EVAL $m  model=${MODEL[$m]}  $(date) ==="
  EXTRA="trainer.val_only=true actor_rollout_ref.rollout.val_kwargs.do_sample=false
         trainer.resume_mode=disable trainer.default_local_dir=$CK/greedy_eval/tmp_${m}
         trainer.project_name=actor_curator_eval trainer.experiment_name=greedy_math_${m}
         actor_rollout_ref.model.path=${MODEL[$m]}" \
  BENCH=math METHOD=uniform NGPU=1 bash repro/run.sh > repro/logs/greedy_math_${m}.log 2>&1 || echo "!! eval failed for $m (see repro/logs/greedy_math_${m}.log)"
  grep -A4 'Initial validation metrics' repro/logs/greedy_math_${m}.log | sed 's/^.*TaskRunner pid=[0-9]*)[^ ]* //'
done
echo "=== SUMMARY (greedy, val-core/*/reward/mean@1) ==="
python - <<'PY'
import glob, re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for d in sorted(glob.glob("tensorboard_log/actor_curator_eval/greedy_math_*")):
    ea = EventAccumulator(d); ea.Reload()
    vals = {t.split('/')[1]: ea.Scalars(t)[-1].value*100 for t in ea.Tags()["scalars"] if t.startswith("val-core/") and t.endswith("reward/mean@1")}
    print(f"{d.split('/')[-1]:24s} " + "  ".join(f"{k}={v:5.1f}" for k, v in sorted(vals.items())))
PY
echo "=== DONE greedy eval $(date) ==="
