#!/bin/bash
#SBATCH --job-name=ac-repro
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --account=raul
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --time=2-12:00:00
#SBATCH --output=repro/logs/%x-%j.out
#SBATCH --error=repro/logs/%x-%j.out
#
# One (benchmark, method) cell of the paper's Table 1 with Qwen2.5-3B actor + Qwen3-0.6B curator.
#   BENCH  = countdown | zebra | arc | math
#   METHOD = ac | uniform | sec | pcl
#   SMOKE=1  -> tiny settings, just checks the pipeline end-to-end
#   EXTRA    -> additional Hydra overrides appended last
# The recipe script is copied to a temp file and patched with sed (MODEL_PATH, WORKING_DIR, GPUS, data files,
# step counts); everything else is a Hydra override. The repo's recipe/ and verl/ files are never modified.
set -euo pipefail

REPO=/shared/home/kseniia.kuvshinova/actor-curator
cd "$REPO"
source /shared/home/kseniia.kuvshinova/miniconda3/etc/profile.d/conda.sh
conda activate verl

export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=/shared/home/kseniia.kuvshinova/.cache/huggingface/datasets   # shared cache chowns files to shared-data -> chmod fails
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1          # single node, no IB on titan
export NCCL_P2P_LEVEL=NVL
unset NCCL_IB_HCA UCX_NET_DEVICES || true
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL || true   # SLURM sets these next to CUDA_VISIBLE_DEVICES; verl's worker refuses both

BENCH=${BENCH:?set BENCH}
METHOD=${METHOD:?set METHOD}
SEED=${SEED:-1}
NGPU=${NGPU:-2}          # 2 = README setup (2 GPUs); 1 = one whole card (shard:10), same batch sizes, TP=1
SMOKE=${SMOKE:-0}
EXTRA=${EXTRA:-}

case "$METHOD" in
  ac)      SCRIPT=recipe/deepscaler/${BENCH}_perf_diff.sh ;;
  uniform) SCRIPT=recipe/deepscaler/${BENCH}_baseline.sh ;;
  sec)     SCRIPT=recipe/deepscaler/countdown_topk.sh ;;        # only clean 3B SEC script (see PROBLEMS.md B6)
  pcl)     if [ "$BENCH" = countdown ] || [ "$BENCH" = math ]; then SCRIPT=recipe/deepscaler/countdown_ps.sh;
           else SCRIPT=recipe/deepscaler/${BENCH}_ps_value.sh; fi ;;
  *) echo "unknown METHOD $METHOD"; exit 1 ;;
esac
case "$BENCH" in
  countdown|zebra|arc) TRAIN=${REPO}/data/combined/train_${BENCH}.parquet; TEST=${REPO}/data/combined/test_${BENCH}.parquet ;;
  math)                TRAIN=${REPO}/data/math/math_train.parquet;        TEST=${REPO}/data/math/math_test.parquet ;;
  *) echo "unknown BENCH $BENCH"; exit 1 ;;
esac

EXP="${BENCH}_${METHOD}_qwen2.5-3b_s${SEED}"
[ "$SMOKE" = 1 ] && EXP="smoke_${EXP}"
CKPT_DIR=/scratch/${USER}/actor-curator-ckpt/${EXP}
mkdir -p "$CKPT_DIR" repro/tmp
TMP=repro/tmp/${EXP}.sh
cp "$SCRIPT" "$TMP"

# --- patch script-level shell variables (keeps every Hydra key unique) ---
sed -i -E \
  -e 's#^MODEL_PATH=.*#MODEL_PATH=Qwen/Qwen2.5-3B#' \
  -e "s#^WORKING_DIR=.*#WORKING_DIR=${REPO}#" \
  -e "s#^export GPUS_PER_NODE=.*#export GPUS_PER_NODE=${NGPU}#" \
  -e "s#^train_files=.*#train_files=${TRAIN}#" \
  -e "s#^test_files=.*#test_files=${TEST}#" \
  -e "s#^exp_name=.*#exp_name=\"${EXP}\"#" \
  -e "s#^project_name=.*#project_name='actor_curator_repro'#" \
  -e 's#^test_freq=.*#test_freq=10#' \
  -e 's#^save_freq=.*#save_freq=20#' \
  -e 's#^total_training_steps=.*#total_training_steps=100#' \
  -e 's#^total_epochs=.*#total_epochs=100000#' \
  -e 's#^val_before_train=.*#val_before_train=true#' \
  -e 's#trainer\.resume_mode=[a-z]+#trainer.resume_mode=auto#' \
  -e 's#^set -xeuo pipefail#set -euo pipefail#' \
  "$TMP"
[ "$NGPU" = 1 ] && sed -i -E 's#^gen_tp=.*#gen_tp=1#' "$TMP"

# --- Hydra overrides common to every run ---
OVR=(
  trainer.default_local_dir="${CKPT_DIR}"
  trainer.max_actor_ckpt_to_keep=1
  trainer.max_critic_ckpt_to_keep=1
  actor_rollout_ref.actor.optim.lr_warmup_steps=25
  ++data.seed=${SEED}
)
# single card: actor+curator+vLLM all share one 96 GB card, ~15 GB are resident at vLLM start -> 0.85 does not fit
[ "$NGPU" = 1 ] && OVR+=( actor_rollout_ref.rollout.gpu_memory_utilization=0.75 )
# --- paper Table 2 values for the curator (PROBLEMS.md B1-B3) ---
case "$METHOD" in
  ac)  OVR+=( ++adv_predictor.dormant_steps=20 ++adv_predictor.critic_warmup=5
              ++adv_predictor.temperature=1.0 ++adv_predictor.temperature_annealing=false
              ++adv_predictor.top_p=0.9 ++adv_predictor.top_p_annealing=false
              ++adv_predictor.perf_diff_amplifier=1000.0 ++critic.clip_range=0.2 ) ;;
  sec) OVR+=( ++adv_predictor.dormant_steps=20 ++adv_predictor.critic_warmup=5 ++critic.clip_range=0.2
              ++adv_predictor.temperature=1.0 ) ;;   # countdown_topk.sh omits it but the trainer reads it unconditionally (unused by stochastic_topk)
  pcl) OVR+=( ++adv_predictor.dormant_steps=20 ++adv_predictor.critic_warmup=5
              ++adv_predictor.top_p=0.9 ++adv_predictor.top_p_annealing=false
              ++adv_predictor.target=abs_adv ) ;;   # *_ps*.sh omit it but the trainer reads it unconditionally (README: PCL target = abs_adv)
esac

if [ "$SMOKE" = 1 ]; then
  sed -i -E \
    -e 's#^total_training_steps=.*#total_training_steps=3#' \
    -e 's#^test_freq=.*#test_freq=3#' \
    -e 's#^save_freq=.*#save_freq=-1#' \
    -e 's#^candidate_batch_size=.*#candidate_batch_size=64#' \
    -e 's#^train_batch_size=.*#train_batch_size=16#' \
    -e 's#^critic_train_batch_size=.*#critic_train_batch_size=16#' \
    -e 's#^n_resp_per_prompt=.*#n_resp_per_prompt=4#' \
    -e 's#^ppo_mini_batch_size=.*#ppo_mini_batch_size=8#' \
    -e 's#^max_response_length=.*#max_response_length=512#' \
    -e "s#^test_files=.*#test_files=${REPO}/repro/tmp/smoke_test_${BENCH}.parquet#" \
    "$TMP"
  OVR+=( ++adv_predictor.dormant_steps=1 ++adv_predictor.critic_warmup=0 actor_rollout_ref.actor.optim.lr_warmup_steps=1 )
fi

if [ "${DRYRUN:-0}" = 1 ]; then echo "--- $EXP ($SCRIPT) patched lines:"; diff "$SCRIPT" "$TMP" | grep "^>" || true; echo "--- overrides: ${OVR[*]} ${EXTRA}"; exit 0; fi
echo "=== $EXP  script=$SCRIPT  NGPU=$NGPU  node=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}  $(date) ==="
echo "overrides: ${OVR[*]} ${EXTRA}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
# one retry if the launch dies within 10 min (seen once: Ray init "RPC error: Deadline Exceeded" when a job starts
# on a node right after the previous job's Ray cluster was still shutting down); a real crash later is not retried.
for attempt in 1 2; do
  t0=$(date +%s)
  # shellcheck disable=SC2086
  bash "$TMP" "${OVR[@]}" ${EXTRA} && break
  rc=$?
  if [ "$attempt" = 1 ] && [ $(( $(date +%s) - t0 )) -lt 600 ]; then echo "!! launch failed early (rc=$rc) — retrying in 60 s"; sleep 60; else exit "$rc"; fi
done
echo "=== DONE $EXP $(date) ==="
