#!/usr/bin/env bash
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda create -n loc2 python=3.10 -y
conda activate loc2

pip install --no-cache-dir --upgrade setuptools==69.5.1 pip pathtools promise pybind11 omegaconf
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu118
pip install --no-cache-dir -r requirements.txt
