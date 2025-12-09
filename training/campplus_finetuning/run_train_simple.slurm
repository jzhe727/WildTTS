#!/bin/bash
#SBATCH --job-name=campplus_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-l40,gpu-a100,gpu-h100


# Campplus Finetuning Training Job
# This runs the simple training pipeline with cosine similarity loss
kill_children() {
    echo "Terminating child processes..."
    pkill --signal SIGTERM --parent $$
}
trap 'kill_children' EXIT

# Activate virtual environment
cd ~/ensf619/
# make subprocess run mlflow_server_script.slurm on the same device
sh ./mlflow_server_script.sh &

sleep 5  # Give some time for the server to start

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate wildtts-torchmetrics

# Set working directory
cd ~/ensf619/WildTTS/training/campplus_finetuning



# Print environment info
echo "=========================================="
echo "Campplus Finetuning Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo "=========================================="

CONFIG="averaged_config_multi_epoch"

# Run training
python train_simple.py \
    --config ${CONFIG}.yaml \
    --embedding_dir ./embedding_artifacts/ \
    --num_workers 16

echo "=========================================="
echo "Training completed at $(date)"
echo "=========================================="

cd ~/ensf619/WildTTS/
python training/campplus_finetuning/create_model_directory.py \
    --source_model_dir backend/CosyVVoice/pretrained_models/CosyVoice2-0.5B/ \
    --model_dir training/campplus_finetuning/checkpoints/${CONFIG}/

sleep 10
echo "Finishing up..."

kill_children
