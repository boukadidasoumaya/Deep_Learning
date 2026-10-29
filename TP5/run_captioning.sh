#!/bin/bash
#SBATCH --job-name=tp5-image-captioning
#SBATCH --account=csc_8607
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/captioning-%j.out
#SBATCH --error=logs/captioning-%j.err

# Se placer dans le dossier du TP
cd ~/TP5

# Utiliser directement le Python de l'environnement deeplearning
PYTHON="/mnt/hdd/homes/sboukadida/miniforge3/envs/deeplearning/bin/python"

# Informations sur le GPU
nvidia-smi

# Lancement de l'entraînement
echo "Début de l'entraînement"
$PYTHON -u train.py
echo "Fin de l'entraînement"
