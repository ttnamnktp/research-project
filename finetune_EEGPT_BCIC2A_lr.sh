#!/bin/bash
#SBATCH --job-name=finetune_EEGPT_BCIC2A_lr
#SBATCH --output=bash_logs/%x_%j.out
#SBATCH --error=bash_logs/%x_%j.err
#SBATCH --time=24:00:00          # hh:mm:ss
#SBATCH --nodes=1                # number of nodes
#SBATCH --gres=gpu:1             # number of GPUs
#SBATCH --partition=P100         # or V100, A100, etc.

set -e  # crash nếu lỗi
set -x  # print command

# ======================
# load environment
# ======================
CONDA_PATH=/home/infres/ttran-25/miniconda3
source "$CONDA_PATH/bin/activate"
conda activate eegpt

# ======================
# kiểm tra GPU
# ======================
nvidia-smi

# ======================
# chạy training
# ======================
srun python -u EEGPT/downstream/finetune_EEGPT_BCIC2A_lr.py