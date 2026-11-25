#!/bin/bash
#SBATCH --job-name=dk1_tasks
#SBATCH --partition=a100-galvani
#SBATCH --account=martius
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/lustre/work/martius/mot244/molmoact2/sim_eval/outputs/slurm-dk1tasks-%j.out

# Bimanual DK1 rolled out zero-shot on the NEW pick-place task suite in ManiSkill,
# driven by the SAME MolmoAct2-BimanualYAM server (DK1 shares YAM kinematics).
# Defaults to the 3 GRASPABLE tasks (Cup/Tool/UtensilTray) — the only ones the
# YAM baseline (job 2621307) showed real partial-credit signal on. OOD / diagnostic.
# Submit:  sbatch sim_eval/scripts/dk1_tasks_sim_eval.sbatch [N_EPISODES]
set -euo pipefail
cd /mnt/lustre/work/martius/mot244/molmoact2

N_EP="${1:-3}"
echo "job ${SLURM_JOB_ID} on $(hostname); GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

POLICY_TYPE=remote-dk1 \
ENVS="${ENVS:-BimanualCup-v1 BimanualTool-v1 BimanualUtensilTray-v1}" \
MAX_STEPS="${MAX_STEPS:-1200}" \
    bash sim_eval/scripts/run_sim_eval_yam_tasks.sh "${N_EP}" rt-fast \
    "sim_eval/outputs/dk1tasks_${SLURM_JOB_ID}"
