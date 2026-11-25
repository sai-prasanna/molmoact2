#!/bin/bash
#SBATCH --job-name=dk1_replay
#SBATCH --partition=a100-galvani
#SBATCH --account=martius
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/lustre/work/martius/mot244/molmoact2/sim_eval/outputs/slurm-replay-%j.out

# Diagnostic: replay a successful YAM trajectory open-loop on DK1 to separate
# execution gap from perception gap. Submit:  sbatch sim_eval/scripts/replay_diag.sbatch [SEED]
set -euo pipefail
cd /mnt/lustre/work/martius/mot244/molmoact2
source .eval_env.sh

SEED="${1:-42}"
PORT=8202
URL="http://127.0.0.1:${PORT}/act"
OUT="sim_eval/outputs/replay_${SLURM_JOB_ID:-local}"
LOG="/tmp/yam_server_${SLURM_JOB_ID:-local}.log"

echo "job ${SLURM_JOB_ID:-local} on $(hostname); seed=${SEED}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== launching MolmoAct2-BimanualYAM server (port ${PORT}) ==="
uv run python examples/yam/host_server_yam.py --host 127.0.0.1 --port "${PORT}" > "${LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

echo "=== waiting for /act health ==="
for i in $(seq 1 240); do
    if curl -sf "${URL}" -o /dev/null 2>/dev/null; then echo "healthy after ${i}0s"; break; fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then echo "!! server died:"; tail -40 "${LOG}"; exit 1; fi
    sleep 10
done
curl -sf "${URL}" -o /dev/null 2>/dev/null || { echo "!! server never healthy:"; tail -40 "${LOG}"; exit 1; }

echo "=== running replay diagnostic ==="
uv run python -m sim_eval.scripts.replay_yam_on_dk1 --url "${URL}" --seed "${SEED}" \
    --max-steps 2000 --attempts 6 --out "${OUT}"
echo "=== done; outputs under ${OUT} ==="
