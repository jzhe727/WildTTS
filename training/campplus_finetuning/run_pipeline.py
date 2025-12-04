#!/usr/bin/env python3
"""
Main Pipeline Script for Campplus Finetuning

This script orchestrates the entire finetuning pipeline:
1. Data preparation (download, DNSMOS scoring, sample selection)
2. Target embedding generation
3. Model training
4. Evaluation with CosyVoice2
5. Upload to HuggingFace

Usage:
    python run_pipeline.py --config config.yaml [--stages all|prepare|embed|train|eval]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import yaml
import mlflow


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*80)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False


def run_data_preparation(config_path: str, output_dir: str) -> bool:
    """Run data preparation stage."""
    cmd = [
        sys.executable,
        "prepare_data.py",
        "--config", config_path,
        "--output_dir", output_dir
    ]
    return run_command(cmd, "Data Preparation")


def run_embedding_generation(config_path: str, data_dir: str, output_dir: str) -> bool:
    """Run target embedding generation stage."""
    cmd = [
        sys.executable,
        "generate_target_embeddings.py",
        "--config", config_path,
        "--data_dir", data_dir,
        "--output_dir", output_dir
    ]
    return run_command(cmd, "Target Embedding Generation")


def run_training(config_path: str, data_dir: str, embedding_dir: str, resume: str = None) -> bool:
    """Run model training stage."""
    cmd = [
        sys.executable,
        "train.py",
        "--config", config_path,
        "--data_dir", data_dir,
        "--embedding_dir", embedding_dir
    ]
    if resume:
        cmd.extend(["--resume", resume])
    return run_command(cmd, "Model Training")


def run_evaluation(config_path: str, checkpoint: str, output_dir: str) -> bool:
    """Run evaluation stage."""
    cmd = [
        sys.executable,
        "evaluate.py",
        "--config", config_path,
        "--checkpoint", checkpoint,
        "--output_dir", output_dir
    ]
    return run_command(cmd, "CosyVoice2 Evaluation")


def setup_mlflow_experiment(config: dict) -> str:
    """Setup MLflow experiment and return run ID."""
    mlflow_config = config['mlflow']
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    mlflow.set_experiment(mlflow_config['experiment_name'])
    
    run_name = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    run = mlflow.start_run(run_name=run_name)
    
    # Log pipeline configuration
    mlflow.log_params({
        'pipeline_run': run_name,
        'dataset': config['data']['huggingface_dataset'],
        'epochs': config['training']['epochs'],
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate']
    })
    
    return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description="Run Campplus finetuning pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        choices=["all", "prepare", "embed", "train", "eval"],
        help="Pipeline stages to run"
    )
    parser.add_argument("--data_dir", type=str, default="./data_artifacts", help="Data artifacts directory")
    parser.add_argument("--embedding_dir", type=str, default="./embedding_artifacts", help="Embedding artifacts directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume/evaluate")
    parser.add_argument("--eval_output", type=str, default="./eval_output", help="Evaluation output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("=" * 80)
    print("Campplus Finetuning Pipeline")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Stages: {args.stages}")
    print(f"Data dir: {args.data_dir}")
    print(f"Embedding dir: {args.embedding_dir}")
    
    # Setup MLflow
    run_id = setup_mlflow_experiment(config)
    print(f"MLflow run ID: {run_id}")
    
    stages_to_run = []
    if args.stages == "all":
        stages_to_run = ["prepare", "embed", "train", "eval"]
    else:
        stages_to_run = [args.stages]
    
    success = True
    
    try:
        # Stage 1: Data Preparation
        if "prepare" in stages_to_run:
            if not run_data_preparation(args.config, args.data_dir):
                success = False
                if args.stages == "all":
                    print("Pipeline stopped due to data preparation failure")
                    return
        
        # Stage 2: Target Embedding Generation
        if "embed" in stages_to_run:
            if not run_embedding_generation(args.config, args.data_dir, args.embedding_dir):
                success = False
                if args.stages == "all":
                    print("Pipeline stopped due to embedding generation failure")
                    return
        
        # Stage 3: Model Training
        if "train" in stages_to_run:
            if not run_training(args.config, args.data_dir, args.embedding_dir, args.checkpoint):
                success = False
                if args.stages == "all":
                    print("Pipeline stopped due to training failure")
                    return
        
        # Stage 4: Evaluation
        if "eval" in stages_to_run:
            checkpoint = args.checkpoint or str(Path(config['training']['checkpoint_dir']) / "best_model.pt")
            if not run_evaluation(args.config, checkpoint, args.eval_output):
                success = False
    
    finally:
        # End MLflow run
        mlflow.log_param("pipeline_success", success)
        mlflow.end_run()
    
    print("\n" + "=" * 80)
    if success:
        print("✓ Pipeline completed successfully!")
    else:
        print("✗ Pipeline completed with errors")
    print("=" * 80)


if __name__ == "__main__":
    main()
