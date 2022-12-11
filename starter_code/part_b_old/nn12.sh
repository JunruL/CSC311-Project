#!/bin/bash
#SBATCH --output=logs_nn12/%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:30:0
#SBATCH --gres=gpu:v100l:1

python nn12.py -lr $1
