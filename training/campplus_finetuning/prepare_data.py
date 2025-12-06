#!/usr/bin/env python3
"""
Data Preparation Script for Campplus Finetuning

This script:
1. Streams TITW dataset from HuggingFace (no local download required)
2. Computes DNSMOS scores for all audio samples using torchmetrics functional API
3. Selects top K samples per speaker based on DNSMOS scores
4. Logs all data artifacts to MLflow

Usage:
    python prepare_data.py --config config.yaml
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Iterator
from collections import defaultdict

import yaml
import numpy as np
import torch
import mlflow
from tqdm import tqdm
from datasets import load_dataset
import librosa
from torchmetrics.functional.audio import deep_noise_suppression_mean_opinion_score


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def stream_dataset_batched(config: dict, subset: str = "easy", batch_size: int = 16) -> Iterator[dict]:
    """
    Stream TITW dataset from HuggingFace without downloading locally.
    """
    dataset_name = config['data']['huggingface_dataset']
    
    print(f"Streaming dataset {dataset_name}/{subset}...")
    
    # Load dataset with streaming enabled
    dataset = load_dataset(
        dataset_name,
        data_dir=subset,
        streaming=True,
        split="train"
    )

    dataset = dataset.batch(batch_size=batch_size)
    
    for sample in dataset:
        yield sample


def compute_dnsmos_scores_streaming(
    config: dict,
    subset: str = "easy",
    batch_size: int = 16
) -> Dict[str, Dict]:
    """
    Compute DNSMOS scores using TorchMetrics functional API.
    """
    
    print("Setting up DNSMOS metric...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_fs = 16000  # DNSMOS expects 16kHz
    
    scores = {}
    
    print(f"Streaming and computing DNSMOS scores for {subset} subset on {device}...")
    
    for batch in tqdm(stream_dataset_batched(config, subset, batch_size=batch_size), desc="Processing batches"):
        audio_tensors = []
        batch_metadata = []
        
        # Pre-process batch
        batch_length = len(batch['id'])
        
        for i in range(batch_length):
            utt_id = batch['id'][i]
            speaker_id = batch['speaker_id'][i]
            text = batch['text'][i]
            
            try:
                audio_data = batch['audio'][i]
                audio_np = np.array(audio_data['array'])
                sr = audio_data['sampling_rate']
                
                # Resample if necessary
                if sr != target_fs:
                    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_fs)
                
                # Normalize audio
                max_val = np.max(np.abs(audio_np))
                if max_val > 0:
                    audio_np = audio_np / max_val
                
                # Convert to tensor
                audio_tensor = torch.from_numpy(audio_np).float()
                
                audio_tensors.append(audio_tensor)
                batch_metadata.append({
                    "utt_id": utt_id,
                    "speaker_id": speaker_id,
                    "text": text
                })
                
            except Exception as e:
                print(f"Warning: Failed to preprocess {utt_id}: {e}")
                scores[utt_id] = {
                    "speaker_id": speaker_id,
                    "text": text,
                    "dnsmos": {"sig": 0.0, "bak": 0.0, "ovr": 0.0}
                }

        if not audio_tensors:
            continue

        # Calculate metrics for the batch
        try:
            # Pad tensors to same length for batching
            input_tensors = [t.to(device) for t in audio_tensors]
            max_len = max(t.shape[0] for t in input_tensors)
            padded_batch = torch.zeros(len(input_tensors), max_len, device=device)
            for idx, t in enumerate(input_tensors):
                padded_batch[idx, :t.shape[0]] = t
            
            # Run prediction using functional API
            # Result is a dictionary containing tensors of scores
            res = deep_noise_suppression_mean_opinion_score(padded_batch, sampling_rate=target_fs, model="v8")
            
            # Iterate through results and map back to metadata
            # The functional output tensors should align with the batch index
            for idx, meta in enumerate(batch_metadata):
                scores[meta['utt_id']] = {
                    "speaker_id": meta['speaker_id'],
                    "text": meta['text'],
                    "dnsmos": {
                        "sig": res['dnsmos_sig'][idx].item(),
                        "bak": res['dnsmos_bak'][idx].item(),
                        "ovr": res['dnsmos_ovr'][idx].item(),
                    }
                }
                
        except Exception as e:
            print(f"Warning: Failed to calculate metrics for batch: {e}")
            # Fallback for failed batch
            for meta in batch_metadata:
                if meta['utt_id'] not in scores:
                    scores[meta['utt_id']] = {
                        "speaker_id": meta['speaker_id'],
                        "text": meta['text'],
                        "dnsmos": {"sig": 0.0, "bak": 0.0, "ovr": 0.0}
                    }
    
    print(f"Computed DNSMOS scores for {len(scores)} samples")
    return scores


def select_top_samples_per_speaker(
    scores_with_metadata: Dict[str, Dict],
    top_k: int = 10
) -> Dict[str, List[Dict]]:
    """
    Select top K samples per speaker based on DNSMOS overall score.
    """
    # Group samples by speaker
    speaker_samples = defaultdict(list)
    
    for utt_id, info in scores_with_metadata.items():
        speaker_samples[info['speaker_id']].append({
            "utt_id": utt_id,
            "text": info['text'],
            "dnsmos": info['dnsmos']
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


def save_artifacts(
    output_dir: Path,
    scores_with_metadata: Dict[str, Dict],
    top_samples: Dict[str, List[Dict]]
):
    """Save all data artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save DNSMOS scores (extract just the scores for backwards compatibility)
    dnsmos_scores = {
        utt_id: info['dnsmos'] 
        for utt_id, info in scores_with_metadata.items()
    }
    with open(output_dir / "dnsmos_scores.json", 'w') as f:
        json.dump(dnsmos_scores, f, indent=2)
    
    # Save full scores with metadata
    with open(output_dir / "scores_with_metadata.json", 'w') as f:
        json.dump(scores_with_metadata, f, indent=2)
    
    # Save top samples per speaker
    with open(output_dir / "top_samples_per_speaker.json", 'w') as f:
        json.dump(top_samples, f, indent=2)
    
    # Save utterance IDs for top samples (for filtering streaming dataset later)
    top_utt_ids = []
    for spk_samples in top_samples.values():
        for sample in spk_samples:
            top_utt_ids.append(sample['utt_id'])
    
    with open(output_dir / "top_utt_ids.json", 'w') as f:
        json.dump(top_utt_ids, f)
    
    print(f"Artifacts saved to {output_dir}")


def log_to_mlflow(config: dict, output_dir: Path, scores_with_metadata: Dict[str, Dict]):
    """Log data preparation artifacts to MLflow."""
    mlflow_config = config['mlflow']
    
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    mlflow.set_experiment(mlflow_config['experiment_name'])
    
    run_name = mlflow_config.get('run_name') or "data_preparation"
    
    with mlflow.start_run(run_name=run_name):
        # Log config as parameters
        mlflow.log_param("huggingface_dataset", config['data']['huggingface_dataset'])
        mlflow.log_param("top_k_samples", config['data']['top_k_samples_per_speaker'])
        
        # Log artifacts
        mlflow.log_artifacts(str(output_dir), artifact_path="data_preparation")
        
        # Log metrics
        avg_ovr = np.mean([info['dnsmos']['ovr'] for info in scores_with_metadata.values()])
        mlflow.log_metric("avg_dnsmos_ovr", avg_ovr)
        mlflow.log_metric("num_samples", len(scores_with_metadata))
        mlflow.log_metric("num_speakers", len(set(info['speaker_id'] for info in scores_with_metadata.values())))
        
        print("Artifacts logged to MLflow")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Campplus finetuning")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--subset", type=str, default="easy", choices=["easy", "hard", "test"],
                        help="Dataset subset to use (easy, hard, or test)")
    parser.add_argument("--output_dir", type=str, default="./data_artifacts", help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("Data Preparation for Campplus Finetuning")
    print("=" * 80)
    
    # Step 1: Stream dataset and compute DNSMOS scores (memory efficient)
    print(f"\nStreaming dataset from HuggingFace ({args.subset} subset) and computing DNSMOS scores...")
    scores_with_metadata = compute_dnsmos_scores_streaming(config, subset=args.subset)
    num_speakers = len(set(info['speaker_id'] for info in scores_with_metadata.values()))
    print(f"Processed {len(scores_with_metadata)} utterances from {num_speakers} speakers")
    
    # Step 2: Select top samples per speaker
    print("\nSelecting top samples per speaker...")
    top_k = config['data']['top_k_samples_per_speaker']
    top_samples = select_top_samples_per_speaker(scores_with_metadata, top_k=top_k)
    print(f"Selected top {top_k} samples for {len(top_samples)} speakers")
    
    # Step 3: Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(output_dir, scores_with_metadata, top_samples)
    
    # Step 4: Log to MLflow
    print("\nLogging to MLflow...")
    log_to_mlflow(config, output_dir, scores_with_metadata)
    
    print("\n" + "=" * 80)
    print("Data preparation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
