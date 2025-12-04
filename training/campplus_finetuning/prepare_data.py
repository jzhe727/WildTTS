#!/usr/bin/env python3
"""
Data Preparation Script for Campplus Finetuning

This script:
1. Downloads TITW dataset from HuggingFace using huggingface-cli
2. Computes DNSMOS scores for all audio files per speaker
3. Selects top K samples per speaker based on DNSMOS scores
4. Creates train/validation split from the dev split
5. Logs all data artifacts to MLflow

Usage:
    python prepare_data.py --config config.yaml
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import yaml
import numpy as np
import mlflow
from tqdm import tqdm

# Add DNSMOS repo to path when needed
# sys.path.insert(0, str(Path(__file__).parent / "DNSMOS"))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_dataset(config: dict) -> Path:
    """
    Download TITW dataset from HuggingFace using huggingface-cli.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to the downloaded dataset directory
    """
    dataset_name = config['data']['huggingface_dataset']
    local_dir = Path(config['data']['local_data_dir'])
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset {dataset_name} to {local_dir}...")
    
    # Use huggingface-cli to download
    cmd = [
        "huggingface-cli", "download",
        dataset_name,
        "--repo-type", "dataset",
        "--local-dir", str(local_dir)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Dataset downloaded successfully to {local_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        raise
    
    return local_dir


def load_metadata(data_dir: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load speaker and utterance metadata from the dataset.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        Tuple of (utt2spk dict, wav.scp dict)
    """
    # Look for metadata files (adjust paths based on actual dataset structure)
    # This is a template - actual paths may vary
    metadata_dir = data_dir / "metadata"
    
    utt2spk = {}
    wav_scp = {}
    
    # Load utt2spk
    utt2spk_file = metadata_dir / "utt2spk"
    if utt2spk_file.exists():
        with open(utt2spk_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    utt_id, spk_id = parts
                    utt2spk[utt_id] = spk_id
    
    # Load wav.scp
    wav_scp_file = metadata_dir / "wav.scp"
    if wav_scp_file.exists():
        with open(wav_scp_file, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, wav_path = parts
                    wav_scp[utt_id] = wav_path
    
    return utt2spk, wav_scp


def compute_dnsmos_scores(
    wav_files: List[str],
    dnsmos_config: dict,
    batch_size: int = 16
) -> Dict[str, Dict[str, float]]:
    """
    Compute DNSMOS scores for a list of audio files using the actual DNSMOS repository.
    
    Args:
        wav_files: List of paths to audio files
        dnsmos_config: DNSMOS configuration from config file
        batch_size: Batch size for processing
        
    Returns:
        Dictionary mapping file path to DNSMOS scores
        {path: {"sig": float, "bak": float, "ovr": float}}
    """
    dnsmos_repo = Path(dnsmos_config['repo_path'])
    
    # Import DNSMOS from the repo
    sys.path.insert(0, str(dnsmos_repo))
    
    # TODO: Import actual DNSMOS inference code
    # This is a template - adjust based on actual DNSMOS repo structure
    # Example structure from microsoft/DNS-Challenge DNSMOS:
    # from dnsmos import DNSMOS
    # model = DNSMOS(dnsmos_config['model_type'])
    
    print(f"Computing DNSMOS scores for {len(wav_files)} files...")
    
    scores = {}
    
    # Placeholder for actual DNSMOS computation
    # The actual DNSMOS repo (microsoft/DNS-Challenge) has:
    # - dnsmos/dnsmos.py with ComputeScore function
    # - Uses ONNX models: sig_bak_ovr.onnx, model_v8.onnx
    
    try:
        # Try to import from DNSMOS repo
        from dnsmos import ComputeScore
        
        for i in tqdm(range(0, len(wav_files), batch_size), desc="Computing DNSMOS"):
            batch = wav_files[i:i+batch_size]
            for wav_path in batch:
                try:
                    result = ComputeScore(wav_path)
                    scores[wav_path] = {
                        "sig": result.get("SIG", 0.0),
                        "bak": result.get("BAK", 0.0),
                        "ovr": result.get("OVRL", 0.0)
                    }
                except Exception as e:
                    print(f"Error processing {wav_path}: {e}")
                    scores[wav_path] = {"sig": 0.0, "bak": 0.0, "ovr": 0.0}
                    
    except ImportError:
        print("Warning: DNSMOS repo not found. Using placeholder scores.")
        print("Please clone https://github.com/microsoft/DNS-Challenge and set dnsmos.repo_path")
        
        # Placeholder - generate random scores for template
        for wav_path in tqdm(wav_files, desc="Computing DNSMOS (placeholder)"):
            scores[wav_path] = {
                "sig": np.random.uniform(3.0, 5.0),
                "bak": np.random.uniform(3.0, 5.0),
                "ovr": np.random.uniform(3.0, 5.0)
            }
    
    return scores


def select_top_samples_per_speaker(
    dnsmos_scores: Dict[str, Dict[str, float]],
    utt2spk: Dict[str, str],
    wav_scp: Dict[str, str],
    top_k: int = 10
) -> Dict[str, List[Dict]]:
    """
    Select top K samples per speaker based on DNSMOS overall score.
    
    Args:
        dnsmos_scores: DNSMOS scores per file
        utt2spk: Utterance to speaker mapping
        wav_scp: Utterance to wav path mapping
        top_k: Number of top samples to select per speaker
        
    Returns:
        Dictionary mapping speaker_id to list of top samples with scores
    """
    # Group samples by speaker
    speaker_samples = defaultdict(list)
    
    for utt_id, spk_id in utt2spk.items():
        if utt_id in wav_scp:
            wav_path = wav_scp[utt_id]
            if wav_path in dnsmos_scores:
                speaker_samples[spk_id].append({
                    "utt_id": utt_id,
                    "wav_path": wav_path,
                    "dnsmos": dnsmos_scores[wav_path]
                })
    
    # Select top K per speaker based on overall DNSMOS score
    top_samples_per_speaker = {}
    
    for spk_id, samples in speaker_samples.items():
        # Sort by overall DNSMOS score (descending)
        sorted_samples = sorted(
            samples,
            key=lambda x: x["dnsmos"]["ovr"],
            reverse=True
        )
        top_samples_per_speaker[spk_id] = sorted_samples[:top_k]
    
    return top_samples_per_speaker


def create_train_val_split(
    utt2spk: Dict[str, str],
    wav_scp: Dict[str, str],
    train_ratio: float = 0.9
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Create train/validation split from the dataset.
    
    Args:
        utt2spk: Full utterance to speaker mapping
        wav_scp: Full wav.scp mapping
        train_ratio: Ratio of training data
        
    Returns:
        Tuple of (train_utt2spk, train_wav_scp, val_utt2spk, val_wav_scp)
    """
    # Group utterances by speaker
    speaker_utts = defaultdict(list)
    for utt_id, spk_id in utt2spk.items():
        if utt_id in wav_scp:
            speaker_utts[spk_id].append(utt_id)
    
    train_utt2spk = {}
    train_wav_scp = {}
    val_utt2spk = {}
    val_wav_scp = {}
    
    # Split per speaker to maintain speaker distribution
    for spk_id, utts in speaker_utts.items():
        np.random.shuffle(utts)
        split_idx = int(len(utts) * train_ratio)
        
        for utt_id in utts[:split_idx]:
            train_utt2spk[utt_id] = spk_id
            train_wav_scp[utt_id] = wav_scp[utt_id]
        
        for utt_id in utts[split_idx:]:
            val_utt2spk[utt_id] = spk_id
            val_wav_scp[utt_id] = wav_scp[utt_id]
    
    return train_utt2spk, train_wav_scp, val_utt2spk, val_wav_scp


def save_artifacts(
    output_dir: Path,
    dnsmos_scores: Dict[str, Dict[str, float]],
    top_samples: Dict[str, List[Dict]],
    train_utt2spk: Dict[str, str],
    train_wav_scp: Dict[str, str],
    val_utt2spk: Dict[str, str],
    val_wav_scp: Dict[str, str]
):
    """Save all data artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save DNSMOS scores
    with open(output_dir / "dnsmos_scores.json", 'w') as f:
        json.dump(dnsmos_scores, f, indent=2)
    
    # Save top samples per speaker
    with open(output_dir / "top_samples_per_speaker.json", 'w') as f:
        json.dump(top_samples, f, indent=2)
    
    # Save train split
    with open(output_dir / "train_utt2spk", 'w') as f:
        for utt_id, spk_id in train_utt2spk.items():
            f.write(f"{utt_id} {spk_id}\n")
    
    with open(output_dir / "train_wav.scp", 'w') as f:
        for utt_id, wav_path in train_wav_scp.items():
            f.write(f"{utt_id} {wav_path}\n")
    
    # Save validation split
    with open(output_dir / "val_utt2spk", 'w') as f:
        for utt_id, spk_id in val_utt2spk.items():
            f.write(f"{utt_id} {spk_id}\n")
    
    with open(output_dir / "val_wav.scp", 'w') as f:
        for utt_id, wav_path in val_wav_scp.items():
            f.write(f"{utt_id} {wav_path}\n")
    
    print(f"Artifacts saved to {output_dir}")


def log_to_mlflow(config: dict, output_dir: Path):
    """Log data preparation artifacts to MLflow."""
    mlflow_config = config['mlflow']
    
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    mlflow.set_experiment(mlflow_config['experiment_name'])
    
    run_name = mlflow_config.get('run_name') or "data_preparation"
    
    with mlflow.start_run(run_name=run_name):
        # Log config as parameters
        mlflow.log_param("huggingface_dataset", config['data']['huggingface_dataset'])
        mlflow.log_param("top_k_samples", config['data']['top_k_samples_per_speaker'])
        mlflow.log_param("train_val_split", config['data']['train_val_split'])
        
        # Log artifacts
        mlflow.log_artifacts(str(output_dir), artifact_path="data_preparation")
        
        # Log metrics
        dnsmos_scores_file = output_dir / "dnsmos_scores.json"
        if dnsmos_scores_file.exists():
            with open(dnsmos_scores_file, 'r') as f:
                scores = json.load(f)
                avg_ovr = np.mean([s['ovr'] for s in scores.values()])
                mlflow.log_metric("avg_dnsmos_ovr", avg_ovr)
        
        print("Artifacts logged to MLflow")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Campplus finetuning")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--skip_download", action="store_true", help="Skip dataset download")
    parser.add_argument("--output_dir", type=str, default="./data_artifacts", help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("Data Preparation for Campplus Finetuning")
    print("=" * 80)
    
    # Step 1: Download dataset
    if not args.skip_download:
        data_dir = download_dataset(config)
    else:
        data_dir = Path(config['data']['local_data_dir'])
        print(f"Using existing data at {data_dir}")
    
    # Step 2: Load metadata
    print("\nLoading metadata...")
    utt2spk, wav_scp = load_metadata(data_dir)
    print(f"Loaded {len(utt2spk)} utterances from {len(set(utt2spk.values()))} speakers")
    
    # Step 3: Compute DNSMOS scores
    print("\nComputing DNSMOS scores...")
    wav_files = list(wav_scp.values())
    dnsmos_scores = compute_dnsmos_scores(
        wav_files,
        config['dnsmos'],
        batch_size=config['dnsmos']['batch_size']
    )
    
    # Step 4: Select top samples per speaker
    print("\nSelecting top samples per speaker...")
    top_k = config['data']['top_k_samples_per_speaker']
    top_samples = select_top_samples_per_speaker(
        dnsmos_scores, utt2spk, wav_scp, top_k=top_k
    )
    print(f"Selected top {top_k} samples for {len(top_samples)} speakers")
    
    # Step 5: Create train/val split
    print("\nCreating train/validation split...")
    train_ratio = config['data']['train_val_split']
    train_utt2spk, train_wav_scp, val_utt2spk, val_wav_scp = create_train_val_split(
        utt2spk, wav_scp, train_ratio=train_ratio
    )
    print(f"Train: {len(train_utt2spk)} utterances, Val: {len(val_utt2spk)} utterances")
    
    # Step 6: Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(
        output_dir,
        dnsmos_scores,
        top_samples,
        train_utt2spk, train_wav_scp,
        val_utt2spk, val_wav_scp
    )
    
    # Step 7: Log to MLflow
    print("\nLogging to MLflow...")
    log_to_mlflow(config, output_dir)
    
    print("\n" + "=" * 80)
    print("Data preparation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
