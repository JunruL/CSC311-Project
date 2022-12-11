#!/bin/bash
#SBATCH --output=logs_nn8/%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=7:30:0
#SBATCH --gres=gpu:v100l:1

python nn8.py -lr $1
