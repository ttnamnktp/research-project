#!/bin/bash
#SBATCH --job-name=linear_probe_CBraMod_BCIC2B_linear_adapter
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
srun python -u CBraMod/finetune_bciciv2b.py \
    --cuda 0 \
    --seed 3407 \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 5e-2 \
    --optimizer AdamW \
    --datasets_dir datasets/downstream/lmdb_2b_0_38Hz \
    --model_dir CBraMod/downstream_checkpoints \
    --downstream_dataset BCIC-IV-2b \
    --frozen True \
    --num_workers 2 \
    --use_adapter True \
    --classifier linear

echo "✅ Job finished"