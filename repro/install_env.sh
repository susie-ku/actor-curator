#!/bin/bash
# Build a conda env "verl" mirroring verl 0.5's cu128 stack (torch 2.7.1 cu128, vLLM 0.10.0, flash-attn 2.8.0.post2)
set -euxo pipefail
source /shared/home/kseniia.kuvshinova/miniconda3/etc/profile.d/conda.sh
conda create -y -n verl python=3.12
conda activate verl
python -m pip install --upgrade pip
pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir vllm==0.10.0
ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
pip install --no-cache-dir "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl"
cd /shared/home/kseniia.kuvshinova/actor-curator
pip install --no-cache-dir -e ".[test,math]"
pip install --no-cache-dir liger-kernel "matplotlib<3.11" "numpy<2" "pyarrow==20.0.0" "scipy<1.16"  # pyarrow>=21 / scipy>=1.16 need numpy>=2, verl pins numpy<2; ray_trainer imports matplotlib
pip uninstall -y pynvml || true
pip install --no-cache-dir --upgrade "nvidia-ml-py>=12.560.30"
python -c "import torch, vllm, flash_attn, transformers, verl; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('vllm', vllm.__version__); print('flash_attn', flash_attn.__version__); print('transformers', transformers.__version__)"
echo INSTALL_DONE
