#!/bin/bash
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=alexnet-full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Se placer dans le dossier TP3
cd "$SLURM_SUBMIT_DIR"

# Chemin de l'environnement Mamba deeplearning
PYTHON="/mnt/hdd/homes/sboukadida/miniforge3/envs/deeplearning/bin/python"

echo "======================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Noeud : $SLURM_NODELIST"
echo "Python : $PYTHON"
echo "Début : $(date)"
echo "======================================"

# Vérifier que Python existe
if [ ! -x "$PYTHON" ]; then
    echo "ERREUR : Python introuvable dans deeplearning"
    exit 1
fi

# Afficher les informations GPU
nvidia-smi

# Vérifier PyTorch et CUDA
"$PYTHON" -c "
import sys
import torch

print('Python :', sys.executable)
print('PyTorch :', torch.__version__)
print('CUDA disponible :', torch.cuda.is_available())

assert torch.cuda.is_available(), 'GPU CUDA indisponible'
"

# Lancer l'entraînement AlexNet
echo "Démarrage de l'entraînement..."
"$PYTHON" -u train_imagenet.py

echo "======================================"
echo "Entraînement terminé : $(date)"
echo "======================================"
