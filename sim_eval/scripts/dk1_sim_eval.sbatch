#!/bin/bash
#SBATCH --job-name=dk1_simeval
#SBATCH --partition=a100-galvani
#SBATCH --account=martius
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/lustre/work/martius/mot244/molmoact2/sim_eval/outputs/slurm-dk1-%j.out

# MolmoAct2-YAM policy driving the bimanual DK1 in ManiSkill (zero-shot).
# Submit:  sbatch sim_eval/scripts/dk1_sim_eval.sbatch [N_EPISODES]
set -euo pipefail
cd /mnt/lustre/work/martius/mot244/molmoact2

N_EP="${1:-10}"
echo "job ${SLURM_JOB_ID} on $(hostname); GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash sim_eval/scripts/run_sim_eval_dk1.sh "${N_EP}" rt-fast \
    "sim_eval/outputs/dk1_${SLURM_JOB_ID}"
