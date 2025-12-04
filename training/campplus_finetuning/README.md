# Campplus Finetuning

This directory contains scripts for finetuning the Campplus speaker embedding model on the TITW dataset.

## Overview

The pipeline consists of the following stages:

1. **Data Preparation** (`prepare_data.py`): Downloads TITW dataset from HuggingFace, computes DNSMOS scores for all audio files, and selects the top K highest-quality samples per speaker.

2. **Target Embedding Generation** (`generate_target_embeddings.py`): Uses the pretrained campplus.onnx model to generate speaker embeddings for the best samples. These serve as target ground truth for training.

3. **Model Training** (`train.py`): Finetunes a Campplus-like model where the input is audio and the target is a randomly selected speaker embedding from the high-quality samples.

4. **Evaluation** (`evaluate.py`): Evaluates CosyVoice2 TTS using the finetuned model for speaker embedding extraction.

## Files

```
campplus_finetuning/
├── config.yaml                    # Main configuration file
├── prepare_data.py                # Data preparation script
├── generate_target_embeddings.py  # Target embedding generation
├── dataset.py                     # PyTorch Dataset and DataLoader
├── model.py                       # Campplus model architecture
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── run_pipeline.py                # Main pipeline orchestrator
└── README.md                      # This file
```

## Requirements

```bash
pip install torch torchaudio numpy pyyaml mlflow tqdm onnxruntime huggingface_hub
```

For DNSMOS scoring, clone the Microsoft DNSMOS repository:
```bash
git clone https://github.com/microsoft/DNS-Challenge DNSMOS
```

## Usage

### Quick Start (Full Pipeline)

```bash
python run_pipeline.py --config config.yaml --stages all
```

### Run Individual Stages

```bash
# 1. Prepare data
python prepare_data.py --config config.yaml --output_dir ./data_artifacts

# 2. Generate target embeddings
python generate_target_embeddings.py --config config.yaml --data_dir ./data_artifacts --output_dir ./embedding_artifacts

# 3. Train model
python train.py --config config.yaml --data_dir ./data_artifacts --embedding_dir ./embedding_artifacts

# 4. Evaluate
python evaluate.py --config config.yaml --checkpoint ./checkpoints/best_model.pt --output_dir ./eval_output
```

## Configuration

All settings are in `config.yaml`. Key parameters:

### Data Settings
- `huggingface_dataset`: HuggingFace dataset path (default: `jungee/titw`)
- `top_k_samples_per_speaker`: Number of best samples to select per speaker (default: 10)
- `train_val_split`: Training/validation split ratio (default: 0.9)

### Training Settings
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 1e-4)
- `loss`: Loss function (default: `cosine_embedding_loss`)

### MLflow Settings
- `tracking_uri`: MLflow tracking server URI
- `experiment_name`: Experiment name for organizing runs

### HuggingFace Settings
- `model_repo`: Repository to upload trained model
- `push_to_hub`: Whether to upload to HuggingFace Hub

## MLflow Tracking

All artifacts and metrics are logged to MLflow:

- **Data Preparation**: DNSMOS scores, top samples list
- **Embeddings**: Target speaker embeddings
- **Training**: Loss curves, learning rate, checkpoints
- **Evaluation**: UTMOS, DNSMOS, speaker similarity, WER

Start the MLflow server:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

## Architecture

The finetuning approach:
1. For each audio sample, extract Fbank features
2. Pass through the Campplus-like encoder
3. Compare output embedding with target embedding (from best samples)
4. Minimize cosine embedding loss

This teaches the model to produce embeddings that match the "ideal" speaker representation derived from high-quality audio samples.

## Outputs

After training:
- `checkpoints/best_model.pt`: Best model checkpoint
- `checkpoints/training_history.json`: Training metrics over epochs
- `eval_output/`: Evaluation results with VERSA metrics

The trained model is also:
- Logged to MLflow as an artifact
- Uploaded to HuggingFace Hub (if enabled)
