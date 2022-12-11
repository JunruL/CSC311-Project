#!/bin/bash
#SBATCH --output=logs_nn10/%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:30:0
#SBATCH --gres=gpu:v100l:1

python nn10.py -lr $1
