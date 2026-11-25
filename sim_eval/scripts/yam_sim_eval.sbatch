#!/bin/bash
#SBATCH --job-name=yam_simeval
#SBATCH --partition=a100-galvani
#SBATCH --account=martius
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/lustre/work/martius/mot244/molmoact2/sim_eval/outputs/slurm-yam-%j.out

# MolmoAct2-YAM policy driving its OWN bimanual YAM in ManiSkill (in-distribution
# control for the DK1 cross-embodiment experiment).
# Submit:  sbatch sim_eval/scripts/yam_sim_eval.sbatch [N_EPISODES]
set -euo pipefail
cd /mnt/lustre/work/martius/mot244/molmoact2

N_EP="${1:-10}"
echo "job ${SLURM_JOB_ID} on $(hostname); GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash sim_eval/scripts/run_sim_eval_yam.sh "${N_EP}" rt-fast \
    "sim_eval/outputs/yam_${SLURM_JOB_ID}"
