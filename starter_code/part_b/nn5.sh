#!/bin/bash
#SBATCH --output=logs_nn5/%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=7:30:0
#SBATCH --gres=gpu:v100l:1

python nn5.py -lr $1
