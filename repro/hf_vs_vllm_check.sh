#!/bin/bash
#SBATCH --job-name=ac-hfvllm
#SBATCH --gres=shard:5
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=raul
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --exclude=mbz-titan-3
#SBATCH --output=repro/logs/%x-%j.out
source /shared/home/kseniia.kuvshinova/miniconda3/etc/profile.d/conda.sh
conda activate verl
export HF_HOME=/shared/models/huggingface HF_HUB_CACHE=/shared/models/huggingface/hub HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=/shared/home/kseniia.kuvshinova/.cache/huggingface/datasets
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
cd /shared/home/kseniia.kuvshinova/actor-curator
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
GMU=0.3 python -W ignore repro/hf_vs_vllm_check.py 2>&1 | grep -v -E "FutureWarning|np.uintc"
echo HFVLLM_CHECK_DONE
