#!/bin/bash
#SBATCH --job-name=yam_tasks
#SBATCH --partition=a100-galvani
#SBATCH --account=martius
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/lustre/work/martius/mot244/molmoact2/sim_eval/outputs/slurm-yamtasks-%j.out

# MolmoAct2-BimanualYAM policy rolled out zero-shot on the NEW pick-place task
# suite (8 envs) in ManiSkill. OOD / diagnostic — see run_sim_eval_yam_tasks.sh.
# Submit:  sbatch sim_eval/scripts/yam_tasks_sim_eval.sbatch [N_EPISODES]
set -euo pipefail
cd /mnt/lustre/work/martius/mot244/molmoact2

N_EP="${1:-3}"
echo "job ${SLURM_JOB_ID} on $(hostname); GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

bash sim_eval/scripts/run_sim_eval_yam_tasks.sh "${N_EP}" rt-fast \
    "sim_eval/outputs/yamtasks_${SLURM_JOB_ID}"
