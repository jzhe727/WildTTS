#!/bin/bash
#SBATCH --job-name=campplus_finetune
#SBATCH --output=logs/campplus_%j.out
#SBATCH --error=logs/campplus_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Campplus Finetuning SLURM Job Script
# =====================================

# Exit on error
set -e

# Create logs directory
mkdir -p logs

# Print job info
echo "=========================================="
echo "Campplus Finetuning Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load modules (adjust based on your cluster)
# module load python/3.10
# module load cuda/11.8

# Activate virtual environment
source ~/venv/bin/activate  # Adjust path as needed

# Navigate to script directory
cd /home/john.zheng1/ensf619/WildTTS/training/campplus_finetuning

# Install requirements if needed
# pip install -r requirements.txt

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Stage to run (can be: all, prepare, embed, train, eval)
STAGE="${1:-all}"

echo "Running stage: $STAGE"
echo ""

# Run the pipeline
python run_pipeline.py \
    --config config.yaml \
    --stages "$STAGE" \
    --data_dir ./data_artifacts \
    --embedding_dir ./embedding_artifacts \
    --eval_output ./eval_output

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="
