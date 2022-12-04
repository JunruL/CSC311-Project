#!/bin/bash
#SBATCH --output=logs_test/%j.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=3:30:0
#SBATCH --gres=gpu:v100l:1

python test.py
