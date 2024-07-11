#!/bin/bash

#SBATCH --job-name vqloc_training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH --partition batch_grad
#SBATCH -w  ariel-v4
#SBATCH -t 3-0
#SBATCH -e slurm/logs/slurm-%A_%x.err
#SBATCH -o slurm/logs/slurm-%A_%x.out

date +%Y-%m-%d/%H:%M:%S

NUMBER_Of_GPUS=1
export base_lr=0.0003
export base_bsz=24
export bsz=$(( $NUMBER_Of_GPUS * 16 ))
export lr=$(python -c "print(f'{""$base_lr / $base_bsz * $bsz"":.5e}')")
echo "Base LR: $base_lr, Base BSZ: $base_bsz, LR: $lr, BSZ: $bsz"

python -m torch.distributed.launch --master_port 9999 --nproc_per_node=8 \
train_anchor.py --cfg ./config/train.yaml

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 9999 --nproc_per_node=1 \
# train_anchor.py --cfg ./config/train.yaml