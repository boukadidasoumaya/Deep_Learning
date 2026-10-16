#!/bin/bash
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -J music-lstm-large
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PYTHON="/mnt/hdd/homes/sboukadida/miniforge3/envs/deeplearning/bin/python"

echo "Job $SLURM_JOB_ID sur $SLURM_NODELIST"
nvidia-smi

"$PYTHON" -u train_music.py
