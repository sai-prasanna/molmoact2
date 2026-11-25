#!/bin/bash
# Run the MolmoAct2-BimanualYAM inference server + the NEW pick-place task suite
# (8 envs) in ManiSkill on one GPU node. Mirrors run_sim_eval_yam.sh's server
# launch but evaluates the full task suite instead of the single legacy task.
# These rollouts are OUT-OF-DISTRIBUTION (real-image-trained policy on sim renders)
# and DIAGNOSTIC ONLY — read n_in_box/success as relative signal, not absolute skill.
# Usage (inside an srun/sbatch GPU allocation):
#   bash sim_eval/scripts/run_sim_eval_yam_tasks.sh [N_EPISODES] [SHADER] [OUTPUT_DIR]
#   ENVS="BimanualBox-v1 BimanualToy-v1" bash sim_eval/scripts/run_sim_eval_yam_tasks.sh 3
set -euo pipefail
cd /mnt/lustre/work/martius/mot244/molmoact2
source .eval_env.sh

N_EP="${1:-3}"
SHADER="${2:-rt-fast}"
OUTPUT_DIR="${3:-sim_eval/outputs/yamtasks_${SLURM_JOB_ID:-local}}"
ENVS="${ENVS:-BimanualBox-v1 BimanualToy-v1 BimanualCup-v1 BimanualMed-v1 BimanualTool-v1 BimanualGrocery-v1 BimanualBlockStack-v1 BimanualUtensilTray-v1}"
# POLICY_TYPE selects the robot embodiment (remote-yam | remote-dk1). Both hit the
# same MolmoAct2-BimanualYAM server below; remote-dk1 only changes robot_uids.
POLICY_TYPE="${POLICY_TYPE:-remote-yam}"
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

echo "=== running YAM task-suite eval: ${N_EP} ep/task, shader=${SHADER}, out=${OUTPUT_DIR} ==="
echo "=== envs: ${ENVS} ==="
uv run python -m sim_eval.run_eval \
    --policy-type "${POLICY_TYPE}" \
    --remote-url "${URL}" \
    -e ${ENVS} \
    -n "${N_EP}" \
    --max-episode-steps "${MAX_STEPS:-1200}" \
    --shader-pack "${SHADER}" \
    --output-dir "${OUTPUT_DIR}"

echo "=== eval done; results under ${OUTPUT_DIR} ==="
