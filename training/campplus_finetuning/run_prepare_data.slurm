#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-l40

# Campplus Finetuning SLURM Job Script
# =====================================

# Activate virtual environment
cd ~/ensf619/
# make subprocess run mlflow_server_script.slurm on the same device
sh -c ./mlflow_server_script.sh 
sleep 20  # Give some time for the server to start

cd ~/ensf619/WildTTS/training/campplus_finetuning
eval "$(conda shell.bash hook)"
conda activate wildtts-torchmetrics

python run_pipeline.py 