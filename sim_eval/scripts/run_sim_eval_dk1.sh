#!/bin/bash
# Run the MolmoAct2-BimanualYAM inference server + DK1 ManiSkill sim eval on one
# GPU node. DK1 is driven zero-shot by the YAM server (shared kinematics).
# Usage (inside an srun/sbatch GPU allocation):
#   bash sim_eval/scripts/run_sim_eval_dk1.sh [N_EPISODES] [SHADER] [OUTPUT_DIR]
set -euo pipefail
cd /mnt/lustre/work/martius/mot244/molmoact2
source .eval_env.sh

N_EP="${1:-10}"
SHADER="${2:-rt-fast}"
OUTPUT_DIR="${3:-sim_eval/outputs/dk1_${SLURM_JOB_ID:-local}}"
PORT=8202
URL="http://127.0.0.1:${PORT}/act"
SERVER_LOG="/tmp/yam_server_${SLURM_JOB_ID:-local}.log"

echo "=== launching MolmoAct2-BimanualYAM server (port ${PORT}) ==="
uv run python examples/yam/host_server_yam.py --host 127.0.0.1 --port "${PORT}" \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'echo "stopping server ${SERVER_PID}"; kill ${SERVER_PID} 2>/dev/null || true' EXIT

echo "=== waiting for /act health (model load + warmup) ==="
for i in $(seq 1 240); do
    if curl -sf "${URL}" -o /dev/null 2>/dev/null; then
        echo "server healthy after ${i}0s"; break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "!! server process died; last log:"; tail -40 "${SERVER_LOG}"; exit 1
    fi
    sleep 10
done
if ! curl -sf "${URL}" -o /dev/null 2>/dev/null; then
    echo "!! server never became healthy; last log:"; tail -40 "${SERVER_LOG}"; exit 1
fi

echo "=== running DK1 sim eval: ${N_EP} episodes, shader=${SHADER}, out=${OUTPUT_DIR} ==="
uv run python -m sim_eval.run_eval \
    --policy-type remote-dk1 \
    --remote-url "${URL}" \
    -e DK1PutEverythingInBox-v1 \
    -n "${N_EP}" \
    --max-episode-steps "${MAX_STEPS:-1600}" \
    --shader-pack "${SHADER}" \
    --output-dir "${OUTPUT_DIR}"

echo "=== eval done; results under ${OUTPUT_DIR} ==="
