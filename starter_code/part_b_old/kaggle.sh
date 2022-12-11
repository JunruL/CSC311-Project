#!/bin/bash
#SBATCH --output=logs_kaggle/%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:30:0
#SBATCH --gres=gpu:v100l:1

python $1 -lr $2 -lamb $3 -num_epoch $4
