#!/usr/bin/env python3
"""
Generate Target Embeddings for Campplus Finetuning

This script:
1. Loads the top samples per speaker (from prepare_data.py output)
2. Uses the pretrained campplus.onnx to generate embeddings for each top sample
3. Saves embeddings as target ground truth for finetuning
4. Logs embeddings to MLflow

The target embeddings serve as the ground truth that the finetuned model
should learn to produce for any audio from that speaker.

Usage:
    python generate_target_embeddings.py --config config.yaml --data_dir ./data_artifacts
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
import numpy as np
import mlflow
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_fbank_features(audio_path: str, config: dict) -> torch.Tensor:
    """
    Extract Fbank features from audio file.
    
    Args:
        audio_path: Path to audio file
        config: Campplus configuration
        
    Returns:
        Fbank features tensor
    """
    # Load audio
    speech, sample_rate = torchaudio.load(audio_path)
    
    # Resample to target sample rate if needed
    target_sr = config.get('sample_rate', 16000)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        speech = resampler(speech)
    
    # Ensure mono
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    
    # Extract Fbank features
    feat = kaldi.fbank(
        speech,
        num_mel_bins=config.get('num_mel_bins', 80),
        dither=0,
        sample_frequency=target_sr
    )
    
    # Mean normalization
    feat = feat - feat.mean(dim=0, keepdim=True)
    
    return feat


def load_campplus_model(model_path: str) -> onnxruntime.InferenceSession:
    """
    Load Campplus ONNX model.
    
    Args:
        model_path: Path to campplus.onnx
        
    Returns:
        ONNX inference session
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Campplus model not found: {model_path}")
    
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    
    # Use CPU provider
    providers = ["CPUExecutionProvider"]
    
    # Try CUDA if available
    if torch.cuda.is_available():
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except:
            pass
    
    session = onnxruntime.InferenceSession(
        model_path,
        sess_options=option,
        providers=providers
    )
    
    return session


def generate_embedding(
    session: onnxruntime.InferenceSession,
    features: torch.Tensor
) -> np.ndarray:
    """
    Generate speaker embedding using Campplus model.
    
    Args:
        session: ONNX inference session
        features: Fbank features tensor
        
    Returns:
        Speaker embedding as numpy array
    """
    # Add batch dimension
    input_data = features.unsqueeze(0).cpu().numpy()
    
    # Run inference
    embedding = session.run(
        None,
        {session.get_inputs()[0].name: input_data}
    )[0].flatten()
    
    return embedding


def generate_target_embeddings(
    top_samples: Dict[str, List[Dict]],
    campplus_session: onnxruntime.InferenceSession,
    campplus_config: dict,
    data_dir: Path
) -> Dict[str, Dict]:
    """
    Generate target embeddings for all top samples per speaker.
    
    Args:
        top_samples: Top samples per speaker from prepare_data.py
        campplus_session: ONNX session for Campplus model
        campplus_config: Campplus configuration
        data_dir: Base directory for resolving relative paths
        
    Returns:
        Dictionary mapping speaker_id to {
            "embeddings": list of embeddings for top samples,
            "mean_embedding": average embedding for speaker,
            "sample_info": sample metadata
        }
    """
    target_embeddings = {}
    
    for spk_id, samples in tqdm(top_samples.items(), desc="Generating embeddings"):
        speaker_embeddings = []
        sample_info = []
        
        for sample in samples:
            wav_path = sample['wav_path']
            
            # Resolve path relative to data_dir if needed
            if not os.path.isabs(wav_path):
                wav_path = str(data_dir / wav_path)
            
            try:
                # Extract features
                features = extract_fbank_features(wav_path, campplus_config)
                
                # Generate embedding
                embedding = generate_embedding(campplus_session, features)
                
                speaker_embeddings.append(embedding)
                sample_info.append({
                    "utt_id": sample['utt_id'],
                    "wav_path": sample['wav_path'],
                    "dnsmos": sample['dnsmos']
                })
                
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")
                continue
        
        if speaker_embeddings:
            # Compute mean embedding for the speaker
            mean_embedding = np.mean(speaker_embeddings, axis=0)
            
            target_embeddings[spk_id] = {
                "embeddings": [emb.tolist() for emb in speaker_embeddings],
                "mean_embedding": mean_embedding.tolist(),
                "sample_info": sample_info,
                "num_samples": len(speaker_embeddings)
            }
    
    return target_embeddings


def save_embeddings(
    target_embeddings: Dict[str, Dict],
    output_dir: Path
):
    """Save embeddings to disk in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON (for inspection)
    with open(output_dir / "target_embeddings.json", 'w') as f:
        json.dump(target_embeddings, f, indent=2)
    
    # Save as PyTorch tensors (for training)
    embeddings_pt = {}
    for spk_id, data in target_embeddings.items():
        embeddings_pt[spk_id] = {
            "embeddings": torch.tensor(data["embeddings"]),
            "mean_embedding": torch.tensor(data["mean_embedding"]),
            "sample_info": data["sample_info"]
        }
    
    torch.save(embeddings_pt, output_dir / "target_embeddings.pt")
    
    # Save embedding lookup (speaker -> mean embedding) for quick access
    embedding_lookup = {
        spk_id: torch.tensor(data["mean_embedding"])
        for spk_id, data in target_embeddings.items()
    }
    torch.save(embedding_lookup, output_dir / "speaker_embedding_lookup.pt")
    
    print(f"Embeddings saved to {output_dir}")


def log_to_mlflow(
    config: dict,
    target_embeddings: Dict[str, Dict],
    output_dir: Path
):
    """Log embeddings to MLflow."""
    mlflow_config = config['mlflow']
    
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    mlflow.set_experiment(mlflow_config['experiment_name'])
    
    with mlflow.start_run(run_name="target_embeddings"):
        # Log configuration
        mlflow.log_param("campplus_model", config['campplus']['pretrained_path'])
        mlflow.log_param("embedding_dim", config['campplus']['embedding_dim'])
        mlflow.log_param("num_speakers", len(target_embeddings))
        
        # Log statistics
        total_embeddings = sum(d['num_samples'] for d in target_embeddings.values())
        mlflow.log_metric("total_embeddings", total_embeddings)
        
        # Log artifacts
        mlflow.log_artifacts(str(output_dir), artifact_path="target_embeddings")
        
        print("Embeddings logged to MLflow")


def main():
    parser = argparse.ArgumentParser(description="Generate target embeddings for Campplus finetuning")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="./data_artifacts", help="Data artifacts directory")
    parser.add_argument("--output_dir", type=str, default="./embedding_artifacts", help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("Target Embedding Generation for Campplus Finetuning")
    print("=" * 80)
    
    # Load top samples from data preparation
    top_samples_file = data_dir / "top_samples_per_speaker.json"
    if not top_samples_file.exists():
        print(f"Error: {top_samples_file} not found. Run prepare_data.py first.")
        sys.exit(1)
    
    with open(top_samples_file, 'r') as f:
        top_samples = json.load(f)
    
    print(f"Loaded top samples for {len(top_samples)} speakers")
    
    # Load Campplus model
    print("\nLoading Campplus model...")
    campplus_path = config['campplus']['pretrained_path']
    
    # Resolve relative path
    if not os.path.isabs(campplus_path):
        campplus_path = str(Path(__file__).parent / campplus_path)
    
    campplus_session = load_campplus_model(campplus_path)
    print(f"Loaded Campplus model from {campplus_path}")
    
    # Generate embeddings
    print("\nGenerating target embeddings...")
    target_embeddings = generate_target_embeddings(
        top_samples,
        campplus_session,
        config['campplus'],
        data_dir
    )
    
    print(f"Generated embeddings for {len(target_embeddings)} speakers")
    
    # Save embeddings
    print("\nSaving embeddings...")
    save_embeddings(target_embeddings, output_dir)
    
    # Log to MLflow
    print("\nLogging to MLflow...")
    log_to_mlflow(config, target_embeddings, output_dir)
    
    print("\n" + "=" * 80)
    print("Target embedding generation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
