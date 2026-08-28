#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

base_python="$HOME/miniconda3/bin/python"
if [[ ! -x "$base_python" ]]; then
  echo "Expected CUDA-enabled Miniconda Python at $base_python" >&2
  exit 1
fi

if [[ ! -x .gpu-venv/bin/python ]]; then
  "$base_python" -m venv --system-site-packages .gpu-venv
fi

.gpu-venv/bin/python -m pip install \
  -r experiments/architecture_ablation_server_requirements.txt

.gpu-venv/bin/python - <<'PY'
import torch
import torchvision
import matplotlib
import sklearn

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("matplotlib", matplotlib.__version__)
print("scikit-learn", sklearn.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the installed PyTorch build")
print("gpu0", torch.cuda.get_device_name(0))
PY

mkdir -p tea_results/architecture_ablation_v1
printf '%s\n' "$(pwd)/.gpu-venv/bin/python" \
  > tea_results/architecture_ablation_v1/python.path
