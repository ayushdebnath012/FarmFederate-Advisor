#!/usr/bin/env bash
# Wait for the running standard-tier jobs, verify they actually completed, and
# only then start the full tier. "If it goes well" is checked, not assumed: a
# crashed or truncated standard run must not silently trigger a 15-hour full run.
set -u
cd "$HOME/FarmFederate" || exit 1

PY="$HOME/miniconda3/envs/evo2/bin/python"
export PYTHONPATH="$HOME/FarmFederate/.pylibs:$HOME/FarmFederate:$HOME/FarmFederate/experiments"
export PYTHONUNBUFFERED=1
LOG="$HOME/FarmFederate/chain_full.log"

say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
say "chain started; waiting for standard-tier jobs"

# ---- wait for both current jobs to exit ----------------------------------
while pgrep -f 'farm_ablation|tea_federated_all_systems' > /dev/null 2>&1; do
    sleep 120
done
say "standard-tier jobs have exited"

# ---- verify each result before chaining ----------------------------------
check_json() {   # $1 = path, $2 = required top-level sections (comma sep)
    "$PY" - "$1" "$2" <<'EOF'
import json, sys
path, required = sys.argv[1], sys.argv[2].split(",")
try:
    d = json.load(open(path))
except Exception as e:
    print(f"FAIL unreadable: {e}"); sys.exit(1)
missing = [s for s in required if s not in d or not d[s]]
if missing:
    print(f"FAIL missing/empty sections: {missing}"); sys.exit(1)
print("OK " + " ".join(f"{s}={len(d[s])}" for s in required))
EOF
}

FARM_OK=0
if grep -q "^Done\." standard50.log 2>/dev/null; then
    if OUT=$(check_json farm_results_standard_genuine.json \
             "E1_alpha_sweep,E2_client_sweep,E8_baselines"); then
        say "farm_ablation standard verified: $OUT"; FARM_OK=1
    else
        say "farm_ablation standard FAILED verification: $OUT"
    fi
else
    say "farm_ablation standard did not print Done.; not chaining"
fi

SYS_OK=0
if grep -q "^wrote " all_systems50.log 2>/dev/null; then
    if OUT=$(check_json tea_results/federated_all_systems/federated_all_systems.json "runs"); then
        say "all_systems verified: $OUT"; SYS_OK=1
    else
        say "all_systems FAILED verification: $OUT"
    fi
else
    say "all_systems did not finish; not chaining"
fi

# ---- launch the full tier only for what passed ---------------------------
if [ "$FARM_OK" = "1" ]; then
    say "launching farm_ablation FULL tier on GPU1"
    CUDA_VISIBLE_DEVICES=1 setsid nohup "$PY" -c \
      "import sys; sys.path.insert(0,'experiments'); import farm_ablation as fa; \
fa.main(base_py='backend/FarmFederate_Colab_Complete.py', tier='full', \
out='farm_results_full_genuine.json', data_root=None)" \
      > full_farm.log 2>&1 < /dev/null &
fi

if [ "$SYS_OK" = "1" ]; then
    say "launching all_systems 3-seed run on GPU0"
    CUDA_VISIBLE_DEVICES=0 setsid nohup "$PY" \
      experiments/tea_federated_all_systems.py \
      --seeds 0,1,2 --rounds 50 \
      --out tea_results/federated_all_systems_full \
      > full_systems.log 2>&1 < /dev/null &
fi

say "chain done (farm_ok=$FARM_OK systems_ok=$SYS_OK)"
