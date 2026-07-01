#!/bin/bash
#SBATCH --job-name=robometer_eval
#SBATCH --partition=a100-galvani
#SBATCH --account=martius
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/lustre/work/martius/mot244/molmoact2/sim_eval/outputs/slurm-robometer-%j.out

# Robometer-4B reward-model inference over the 30 PutEverythingInBox rollout videos,
# scored against ManiSkill ground truth (success, n_in_box).
# Submit:  sbatch sim_eval/robometer_eval/robometer_eval.sbatch
set -euo pipefail

ROBOMETER=/mnt/lustre/work/martius/mot244/robometer
REPO=/mnt/lustre/work/martius/mot244/molmoact2
OUT="${REPO}/sim_eval/robometer_eval/results"

echo "job ${SLURM_JOB_ID:-local} on $(hostname); GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "${ROBOMETER}"
# Robometer-4B is trained on 8 frames; Unsloth forces eager attention (O(seq^2) mem), and the
# render frames are 720x1280, so 32 frames OOMs a 40GB A100. 8 frames matches training and fits.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python "${REPO}/sim_eval/robometer_eval/run_robometer_batch.py" \
    --repo-root "${REPO}" \
    --out-dir "${OUT}" \
    --fps 8.0 --max-frames 8

echo "=== analysis ==="
cd "${REPO}"
uv run python sim_eval/robometer_eval/analyze.py \
    --scores "${OUT}/scores.json" \
    --out "${OUT}/metrics.json"
