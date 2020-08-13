#!/usr/bin/env bash

#SBATCH --partition=batch_default

#SBATCH --cpus-per-task=1

#SBATCH --gres=gpu:4

./a.out
