#!/usr/bin/env bash

#SBATCH --partition=batch_default

#SBATCH --time=0-0:30:0

#SBATCH --cpus-per-task=1

#SBATCH --gres=gpu:1

./b.out
