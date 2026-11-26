#!/bin/bash
#SBATCH --job-name=ac-rngcheck
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=raul
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --exclude=mbz-titan-3
#SBATCH --output=repro/logs/%x-%j.out
source /shared/home/kseniia.kuvshinova/miniconda3/etc/profile.d/conda.sh
conda activate verl
export HF_HOME=/shared/models/huggingface HF_HUB_CACHE=/shared/models/huggingface/hub HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=/shared/home/kseniia.kuvshinova/.cache/huggingface/datasets
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
cd /shared/home/kseniia.kuvshinova/actor-curator
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
python -W ignore repro/rng_check.py 2>&1 | grep -v -E "FutureWarning|np.uintc"
echo RNG_CHECK_DONE
