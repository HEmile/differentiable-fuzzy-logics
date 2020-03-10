#!/usr/bin/env bash
#SBATCH --time=2-12:15:00
#SBATCH -N 1
#SBATCH -C TitanX
#SBATCH --gres=gpu:1

module add cuda90

source activate thesis
export PYTHONPATH=/home/ekn274/mnist

python main.py "$@"