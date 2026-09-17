#!/usr/bin/env bash
# Run FarmFederate multimodal training on the IIT box reachable through serveo.
#
#   scripts/remote_train.sh setup            # rsync code + weather data, create venv, install deps
#   scripts/remote_train.sh sync             # rsync code only
#   scripts/remote_train.sh train [args...]  # launch backend/multimodal_train.py under nohup
#   scripts/remote_train.sh fed              # launch backend/train_fed_multimodal.py under nohup
#   scripts/remote_train.sh logs [run]       # tail the latest (or named) run log
#   scripts/remote_train.sh status           # GPU + running python processes
#   scripts/remote_train.sh fetch [run]      # copy metrics/history (not weights) back to ./remote_runs/
#   scripts/remote_train.sh ssh [cmd]        # interactive shell / one-off command
#
# Auth: password login only works interactively; run `scripts/remote_train.sh ssh` once and
# append your public key to ~/.ssh/authorized_keys on the box (or use ssh-copy-id) so the
# non-interactive commands here work.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sn4622130673}"
REMOTE_USER="${REMOTE_USER:-iit}"
REMOTE_DIR="${REMOTE_DIR:-~/FarmFederate}"
PROXY="ssh -o StrictHostKeyChecking=accept-new -W ${REMOTE_HOST}:22 serveo.net"
SSH_OPTS=(-o "ProxyCommand=${PROXY}" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
SSH="ssh ${SSH_OPTS[*]} ${REMOTE_USER}@${REMOTE_HOST}"
RSYNC_SSH="ssh -o ProxyCommand='${PROXY}' -o StrictHostKeyChecking=accept-new"

HERE="$(cd "$(dirname "$0")/.." && pwd)"

sync_code() {
  rsync -az --info=progress2 -e "$RSYNC_SSH" \
    --include='backend/' --include='backend/*.py' --include='backend/data/' --include='backend/data/weather/' --include='backend/data/weather/*' \
    --include='Weather data/' --include='Weather data/*.xlsx' \
    --include='tests/' --include='tests/test_weather_data.py' \
    --include='scripts/' --include='scripts/remote_train.sh' \
    --include='requirements.txt' --exclude='*' \
    "$HERE/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
}

case "${1:-}" in
  setup)
    sync_code
    $SSH bash -s <<'REMOTE'
set -e
cd ~/FarmFederate
if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
pip install -q --upgrade pip
if command -v nvidia-smi >/dev/null 2>&1; then
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121 || pip install -q torch torchvision
else
  pip install -q torch torchvision
fi
pip install -q transformers datasets tokenizers numpy pandas openpyxl Pillow scikit-learn tqdm pytest
python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
cd backend && python weather_data.py | head -3
REMOTE
    ;;
  sync) sync_code ;;
  train)
    shift
    sync_code
    $SSH "cd ${REMOTE_DIR}/backend && mkdir -p logs && . ../.venv/bin/activate && \
      RUN=train_\$(date +%Y%m%d_%H%M%S) && \
      nohup python -u multimodal_train.py $* > logs/\$RUN.log 2>&1 & echo launched; sleep 2; ls -t ${REMOTE_DIR}/backend/logs | head -1"
    ;;
  fed)
    shift
    sync_code
    $SSH "cd ${REMOTE_DIR}/backend && mkdir -p logs && . ../.venv/bin/activate && \
      RUN=fed_\$(date +%Y%m%d_%H%M%S) && \
      env $* nohup python -u train_fed_multimodal.py > logs/\$RUN.log 2>&1 & echo launched; sleep 2; ls -t ${REMOTE_DIR}/backend/logs | head -1"
    ;;
  logs)
    RUN="${2:-}"
    $SSH "cd ${REMOTE_DIR}/backend/logs && f=\${RUN:-\$(ls -t | head -1)}; echo \"== \$f\"; tail -n \${LINES:-40} \$f"
    ;;
  status)
    $SSH "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null; ps -eo pid,etime,%cpu,%mem,cmd | grep -E 'multimodal_train|train_fed_multimodal' | grep -v grep || echo 'no training process'"
    ;;
  fetch)
    mkdir -p "$HERE/remote_runs"
    rsync -az -e "$RSYNC_SSH" --include='*/' --include='*.json' --include='*.log' --exclude='*' \
      "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/backend/checkpoints/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/backend/logs/" "$HERE/remote_runs/"
    find "$HERE/remote_runs" -name metrics.json | sort
    ;;
  ssh) shift; $SSH "$@" ;;
  *) sed -n 2,16p "$0"; exit 1 ;;
esac
