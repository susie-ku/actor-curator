#!/bin/bash
#SBATCH --job-name=ac-gpucheck
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=raul
#SBATCH --qos=normal
#SBATCH --partition=debug
#SBATCH --time=00:20:00
#SBATCH --output=repro/logs/%x-%j.out
set -x
source /shared/home/kseniia.kuvshinova/miniconda3/etc/profile.d/conda.sh
conda activate verl
export HF_HOME=/shared/models/huggingface HF_HUB_CACHE=/shared/models/huggingface/hub HF_HUB_OFFLINE=1
nvidia-smi -L
python -W ignore repro/gpu_check.py
echo GPU_CHECK_DONE
