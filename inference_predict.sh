#!/bin/bash

#SBATCH --job-name vqloc_cheating
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH --partition batch_grad
#SBATCH -w  ariel-v5
#SBATCH -t 3-0
#SBATCH -e slurm/logs/slurm-%A_%x.err
#SBATCH -o slurm/logs/slurm-%A_%x.out

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python inference_predict.py --eval --cfg ./config/val.yaml \
 --window_cheating True \
 --window_size 45