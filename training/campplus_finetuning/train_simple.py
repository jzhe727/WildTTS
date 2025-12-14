#!/usr/bin/env python3
"""
Simple Campplus Finetuning Training Script

This script:
1. Streams training data from HuggingFace
2. Loads target embeddings from speaker_embedding_lookup.pt (mean embeddings)
3. Uses cosine similarity loss for finetuning
4. Logs training metrics to MLflow

Usage:
    python train_simple.py --config config.yaml
"""

import os
import sys
import json
import argparse
import subprocess
import random
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Iterator, List

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from datasets import load_dataset
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


from model import create_model
from create_model_directory import create_model_directory

torch.manual_seed(619)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_fbank_features(
    audio_array: np.ndarray,
    sample_rate: int,
    num_mel_bins: int = 80,
    target_sr: int = 16000
) -> torch.Tensor:
    """
    Extract Fbank features from audio array.
    
    Args:
        audio_array: Audio data as numpy array
        sample_rate: Sample rate of audio
        num_mel_bins: Number of mel filter banks
        target_sr: Target sample rate
        
    Returns:
        Fbank features tensor of shape (time, num_mel_bins)
    """
    # Convert to torch tensor
    speech = torch.from_numpy(audio_array).float()
    
    # Add channel dimension if needed
    if speech.dim() == 1:
        speech = speech.unsqueeze(0)
    
    # Resample to target sample rate if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        speech = resampler(speech)
    
    # Ensure mono
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    
    # Normalize audio
    max_val = torch.max(torch.abs(speech))
    if max_val > 0:
        speech = speech / max_val
    
    # Extract Fbank features
    feat = kaldi.fbank(
        speech,
        num_mel_bins=num_mel_bins,
        dither=0,
        sample_frequency=target_sr
    )
    
    # Mean normalization
    feat = feat - feat.mean(dim=0, keepdim=True)
    
    return feat


class HFStreamingDataset(IterableDataset):
    """
    IterableDataset that streams from HuggingFace and processes audio in workers.
    """
    
    def __init__(
        self, 
        config: dict, 
        averaged_speaker_embeddings: Optional[Dict[str, torch.Tensor]] = None, 
        speaker_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        subset: str = "easy",
        averaged: bool = True
    ):
        self.config = config
        self.averaged_speaker_embeddings = averaged_speaker_embeddings
        self.speaker_embeddings = speaker_embeddings
        self.subset = subset
        self.dataset_name = config['data']['huggingface_dataset']
        self.num_mel_bins = config['campplus']['num_mel_bins']
        self.averaged = averaged

        if averaged:
            assert averaged_speaker_embeddings is not None, "Averaged speaker embeddings must be provided if averaged=True"
        else:
            assert speaker_embeddings is not None, "Speaker embeddings must be provided if averaged=False"

        # Load dataset in __init__ to avoid concurrent access issues
        print(f"Loading dataset {self.dataset_name} (subset={self.subset})...")
        self.ds = load_dataset(
            self.dataset_name,
            data_dir=self.subset,
            streaming=True,
            split="train"
        )
        print(f"Dataset loaded, has {self.ds.num_shards} shards.")
        
    def __iter__(self):
        worker_info = get_worker_info()
        
        # NOTE: Do NOT manually shard the dataset here!
        # HuggingFace streaming datasets automatically distribute shards across
        # PyTorch DataLoader workers. Manual sharding causes conflicts and results
        # in the "Too many dataloader workers" warning.
        
        processed_count = 0
        skipped_count = 0
        
        for item in self.ds:
            utt_id = item['id']
            spk_id = item['speaker_id']
            
            # Skip if speaker not in embeddings
            if self.averaged:
                if spk_id not in self.averaged_speaker_embeddings:
                    skipped_count += 1
                    continue
            else:
                if spk_id not in self.speaker_embeddings:
                    skipped_count += 1
                    continue
            
            try:
                audio_data = item['audio']
                audio_array = np.array(audio_data['array'])
                sample_rate = audio_data['sampling_rate']
                
                # Extract features (expensive operation, done in worker)
                features = extract_fbank_features(
                    audio_array, 
                    sample_rate, 
                    self.num_mel_bins
                )
                
                if self.averaged:
                    target_embedding = self.averaged_speaker_embeddings[spk_id]
                else:
                    # select random integer and only take that index through pytorch
                    num_embeddings = self.speaker_embeddings[spk_id].shape[0]
                    random_idx = torch.randint(0, num_embeddings, (1,)).item()
                    target_embedding = self.speaker_embeddings[spk_id][random_idx]

                processed_count += 1
                yield {
                    "features": features,
                    "target_embedding": target_embedding,
                    "utt_id": utt_id,
                    "spk_id": spk_id
                }
                
            except Exception as e:
                print(f"Error processing {utt_id}: {e}")
                continue
                
        if worker_info is not None:
            print(f"Worker {worker_info.id}: Processed {processed_count}, Skipped {skipped_count}")


def collate_fn(batch: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
    """
    Collate function to pad features and stack embeddings.
    """
    # Filter out None items if any (though dataset shouldn't yield None)
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
        
    # Pad features
    lengths = [item['features'].shape[0] for item in batch]
    max_len = max(lengths)
    num_mel_bins = batch[0]['features'].shape[1]
    
    batch_size = len(batch)
    padded_features = torch.zeros(batch_size, max_len, num_mel_bins)
    
    for i, item in enumerate(batch):
        feat = item['features']
        padded_features[i, :feat.shape[0]] = feat
        
    target_embeddings = torch.stack([item['target_embedding'] for item in batch])
    
    return {
        "features": padded_features,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "target_embeddings": target_embeddings,
        "utt_ids": [item['utt_id'] for item in batch],
        "spk_ids": [item['spk_id'] for item in batch]
    }


class CosineSimilarityLoss(nn.Module):
    """
    Cosine similarity loss that maximizes similarity between predictions and targets.
    
    Loss = 1 - cosine_similarity(pred, target)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity loss.
        
        Args:
            predictions: Predicted embeddings (batch, embedding_dim)
            targets: Target embeddings (batch, embedding_dim)
            
        Returns:
            Scalar loss value
        """
        cosine_sim = nn.functional.cosine_similarity(predictions, targets, dim=1)
        loss = 1.0 - cosine_sim.mean()
        return loss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    config: dict,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int = 0,
    prev_steps: int = 0
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader instance
        config: Configuration dict
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    # Freeze BatchNorm for stability with small batches
    model.freeze_batchnorm()
    
    criterion = CosineSimilarityLoss()
    gradient_clip = config['training'].get('gradient_clip', 1.0)
    
    total_loss = 0.0
    total_cosine_sim = 0.0
    num_batches = 0
    num_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")
    
    for batch in pbar:
        if batch is None:
            continue
            
        features = batch['features'].to(device)
        targets = batch['target_embeddings'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        embeddings = model(features)
        
        # Compute loss
        loss = criterion(embeddings, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            cosine_sim = nn.functional.cosine_similarity(embeddings, targets, dim=1).mean()
        
        total_loss += loss.item()
        total_cosine_sim += cosine_sim.item()
        num_batches += 1
        num_samples += features.shape[0]

        mlflow.log_metrics({
            'batch_cosine_similarity': cosine_sim.item(),
            'batch_size': features.shape[0]
        }, step=prev_steps + num_batches)

        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cos_sim': f'{cosine_sim.item():.4f}',
            'samples': num_samples
        })
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_cosine_sim = total_cosine_sim / max(num_batches, 1)
    
    return {
        'loss': avg_loss,
        'cosine_similarity': avg_cosine_sim,
        'num_samples': num_samples,
        'num_batches': num_batches
    }


def export_onnx_model(model: nn.Module, output_path: str, opset_version: int = 14):
    """
    Export the PyTorch model to ONNX format.
    
    Args:
        model: The PyTorch model to export
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version
        
    Note:
        Uses legacy ONNX exporter (not dynamo=True) because onnx2torch-converted
        models contain data-dependent operations that are incompatible with
        torch.export.export.
        
        IMPORTANT: Model must be moved to CPU before export to avoid segfaults.
        onnx2torch-converted models contain operations that call .numpy() or 
        interact with CPU during tracing, which causes segfaults when exporting
        from GPU.
    """
    # Move model to CPU for ONNX export - required for onnx2torch models
    # to avoid segfaults during tracing (some ops call .numpy() internally)
    original_device = next(model.parameters()).device
    model = model.cpu()
    model.eval()
    
    # Create dummy input on CPU
    dummy_input = torch.randn(1, 200, 80)
    
    # Use legacy ONNX exporter - dynamo=True is incompatible with onnx2torch models
    # due to data-dependent operations in gather/reshape nodes
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['speech'],
        output_names=['embedding'],
        dynamic_axes={
            'speech': {0: 'batch_size', 1: 'time'},
            'embedding': {0: 'batch_size'}
        },
        dynamo=False
    )
    print(f"Model exported to {output_path}")
    
    # Move model back to original device if needed
    if original_device != torch.device('cpu'):
        model = model.to(original_device)


def load_titw_metadata(metadata_dir: Path) -> Dict:
    """
    Load TITW test metadata files.
    
    Args:
        metadata_dir: Path to metadata directory
        
    Returns:
        dict: Mapping of utterance_id to {text, wav_path, speaker_id}
    """
    # Load text_test: utterance_id + transcript
    text_dict = {}
    with open(metadata_dir / 'text_test', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                utt_id, text = parts
                text_dict[utt_id] = text
    
    # Load wav_test.scp: utterance_id + relative_wav_path
    wav_dict = {}
    with open(metadata_dir / 'wav_test.scp', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                utt_id, wav_path = parts
                wav_dict[utt_id] = wav_path
    
    # Load utt2spk_test: utterance_id + speaker_id
    spk_dict = {}
    with open(metadata_dir / 'utt2spk_test', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                utt_id, spk_id = parts
                spk_dict[utt_id] = spk_id
    
    # Combine all metadata
    metadata = {}
    for utt_id in text_dict.keys():
        if utt_id in wav_dict and utt_id in spk_dict:
            metadata[utt_id] = {
                'text': text_dict[utt_id],
                'wav_path': wav_dict[utt_id],
                'speaker_id': spk_dict[utt_id]
            }
    
    return metadata


def sample_titw_subset(
    metadata: Dict, 
    num_samples: int = 450,
    output_dir: Path = None,
    seed: int = None
) -> tuple:
    """
    Sample a random subset of TITW test utterances and create subset metadata files.
    
    Args:
        metadata: Full TITW metadata dictionary
        num_samples: Number of samples to select
        output_dir: Directory to save subset metadata files
        seed: Random seed for reproducibility (optional)
        
    Returns:
        tuple: (subset_metadata, metadata_dir) - sampled metadata and path to subset metadata files
    """
    if seed is not None:
        random.seed(seed)
    
    # Sample random utterance IDs
    all_utt_ids = list(metadata.keys())
    num_samples = min(num_samples, len(all_utt_ids))
    sampled_ids = random.sample(all_utt_ids, num_samples)
    
    # Create subset metadata
    subset_metadata = {utt_id: metadata[utt_id] for utt_id in sampled_ids}
    
    # Create metadata files if output_dir provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write text_test
        with open(output_dir / 'text_test', 'w', encoding='utf-8') as f:
            for utt_id in sorted(sampled_ids):
                f.write(f"{utt_id} {metadata[utt_id]['text']}\n")
        
        # Write wav_test.scp
        with open(output_dir / 'wav_test.scp', 'w', encoding='utf-8') as f:
            for utt_id in sorted(sampled_ids):
                f.write(f"{utt_id} {metadata[utt_id]['wav_path']}\n")
        
        # Write utt2spk_test
        with open(output_dir / 'utt2spk_test', 'w', encoding='utf-8') as f:
            for utt_id in sorted(sampled_ids):
                f.write(f"{utt_id} {metadata[utt_id]['speaker_id']}\n")
        
        # Write text file for eval_versa (uses 'text' not 'text_test')
        with open(output_dir / 'text', 'w', encoding='utf-8') as f:
            for utt_id in sorted(sampled_ids):
                f.write(f"{utt_id} {metadata[utt_id]['text']}\n")
    
    return subset_metadata, output_dir


def run_epoch_evaluation(
    model: nn.Module,
    config: dict,
    epoch: int,
    checkpoint_dir: Path,
    titw_metadata: Dict,
    num_eval_samples: int = 450,
    source_model_dir: str = None,
    titw_test_dir: Path = None,
    eval_dir: Path = None
) -> Dict:
    """
    Run end-to-end evaluation after an epoch:
    1. Export model to ONNX
    2. Create mock model directory with finetuned campplus
    3. Generate TTS samples using generate_titw.py (wildtts env)
    4. Evaluate using eval_versa.py (wildtts-versa env)
    5. Log results and artifacts to MLflow
    
    Args:
        model: Trained model
        config: Training config
        epoch: Current epoch number
        checkpoint_dir: Directory for checkpoints
        titw_metadata: Full TITW metadata dictionary
        num_eval_samples: Number of samples to evaluate
        source_model_dir: Path to source CosyVoice2 model directory
        titw_test_dir: Path to TITW test directory
        eval_dir: Path to eval directory
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Running epoch {epoch + 1} evaluation with {num_eval_samples} samples")
    print(f"{'='*80}")
    
    # Create temporary directory for this epoch's evaluation
    epoch_eval_dir = checkpoint_dir / f"eval_epoch_{epoch + 1}"
    epoch_eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Export model to ONNX
    onnx_path = epoch_eval_dir / "campplus_finetuned.onnx"
    print(f"Exporting model to ONNX: {onnx_path}")
    export_onnx_model(model, str(onnx_path))
    
    # Step 2: Create mock model directory with finetuned campplus
    print("Creating model directory with finetuned campplus...")
    create_model_directory(
        model_dir=str(epoch_eval_dir),
        source_model_dir=source_model_dir,
        campplus_onnx_path=str(onnx_path)
    )
    mock_model_dir = epoch_eval_dir / "CosyVoice2-0.5B"
    
    # Step 3: Sample random subset of TITW test utterances
    subset_metadata_dir = epoch_eval_dir / "subset_metadata"
    print(f"Sampling {num_eval_samples} random utterances for evaluation...")
    subset_metadata, _ = sample_titw_subset(
        titw_metadata, 
        num_samples=num_eval_samples,
        output_dir=subset_metadata_dir,
        seed=619  # Same seed every epoch for consistency
    )
    print(f"Sampled {len(subset_metadata)} utterances")
    
    # Step 4: Generate TTS samples using wildtts conda env
    output_gen_dir = epoch_eval_dir / "generated"
    print(f"Generating TTS samples to: {output_gen_dir}")
    
    generate_cmd = [
        "bash", "-c",
        f"eval \"$(conda shell.bash hook)\" && "
        f"conda activate wildtts && "
        f"cd {eval_dir} && "
        f"python generate_titw.py "
        f"--metadata_dir {subset_metadata_dir} "
        f"--test_wav_dir {titw_test_dir} "
        f"--output_dir {output_gen_dir} "
        f"--model_dir {mock_model_dir}"
    ]
    
    print(f"Running generation command...")
    try:
        result = subprocess.run(
            generate_cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        print("Generation completed successfully")
        if result.stdout:
            print(result.stdout[-2000:])  # Last 2000 chars
    except subprocess.CalledProcessError as e:
        print(f"Generation failed with exit code {e.returncode}")
        print(f"STDERR: {e.stderr[-2000:] if e.stderr else 'None'}")
        print(f"STDOUT: {e.stdout[-2000:] if e.stdout else 'None'}")
        return {'error': 'generation_failed', 'epoch': epoch + 1}
    except subprocess.TimeoutExpired:
        print("Generation timed out after 1 hour")
        return {'error': 'generation_timeout', 'epoch': epoch + 1}
    
    # Step 5: Run evaluation using wildtts-versa conda env
    eval_output_dir = epoch_eval_dir / "eval_results"
    eval_config = eval_dir / "configs" / "no_mcd.yaml"
    versa_dir = eval_dir / "versa"
    
    
    print(f"Running VERSA evaluation...")
    eval_cmd = [
        "bash", "-c",
        f"eval \"$(conda shell.bash hook)\" && "
        f"conda activate wildtts-versa && "
        f"cd {eval_dir} && "
        f"python eval_versa.py "
        f"--gt_dir {titw_test_dir} "
        f"--pred_dir {output_gen_dir} "
        f"--output_dir {eval_output_dir} "
        f"--config {eval_config} "
        f"--versa_dir {versa_dir} "
        f"--text {subset_metadata_dir / 'text'}"
    ]
    
    try:
        result = subprocess.run(
            eval_cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout
        )
        print("Evaluation completed successfully")
        if result.stdout:
            print(result.stdout[-2000:])
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit code {e.returncode}")
        print(f"STDERR: {e.stderr[-2000:] if e.stderr else 'None'}")
        print(f"STDOUT: {e.stdout[-2000:] if e.stdout else 'None'}")
        return {'error': 'evaluation_failed', 'epoch': epoch + 1}
    except subprocess.TimeoutExpired:
        print("Evaluation timed out after 30 minutes")
        return {'error': 'evaluation_timeout', 'epoch': epoch + 1}
    
    # Step 6: Parse and return results
    metrics = {'epoch': epoch + 1}
    
    # Load statistics from eval_versa output
    stats_file = eval_output_dir / "statistics.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        for metric_name, metric_values in stats.items():
            if 'mean' in metric_values:
                metrics[f'eval_{metric_name}_mean'] = metric_values['mean']
            if 'min' in metric_values:
                metrics[f'eval_{metric_name}_min'] = metric_values['min']
            if 'max' in metric_values:
                metrics[f'eval_{metric_name}_max'] = metric_values['max']
    
    # Log artifacts to MLflow
    print("Logging evaluation artifacts to MLflow...")
    
    # Log evaluation statistics
    if stats_file.exists():
        mlflow.log_artifact(str(stats_file), f"eval_epoch_{epoch + 1}")
    
    # Log summary if exists
    summary_file = eval_output_dir / "summary.txt"
    if summary_file.exists():
        mlflow.log_artifact(str(summary_file), f"eval_epoch_{epoch + 1}")
    
    # Log all generated audio files
    all_wavs = list(output_gen_dir.glob("*.wav"))
    for wav_file in all_wavs:
        mlflow.log_artifact(str(wav_file), f"eval_epoch_{epoch + 1}/samples")
    
    # Log the ONNX model for this epoch
    mlflow.log_artifact(str(onnx_path), f"eval_epoch_{epoch + 1}")
    
    # Clean up large files to save disk space (keep stats and samples)
    # Remove generated audio except samples
    for wav_file in output_gen_dir.glob("*.wav"):
        wav_file.unlink()
    
    # Remove mock model directory (large)
    shutil.rmtree(mock_model_dir, ignore_errors=True)
    
    print(f"Epoch {epoch + 1} evaluation complete. Metrics: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Simple Campplus Finetuning")
    parser.add_argument("--config", type=str, default="averaged_config.yaml", help="Path to config file")
    parser.add_argument("--embedding_dir", type=str, default="./embedding_artifacts", help="Embedding artifacts directory")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--averaged", type=bool, default=True, help="Use averaged speaker embeddings or not")
    
    # Evaluation arguments
    parser.add_argument("--eval_samples", type=int, default=450, help="Number of samples for epoch evaluation")
    parser.add_argument("--source_model_dir", type=str, 
                        default="../backend/CosyVoice/pretrained_models/CosyVoice2-0.5B",
                        help="Path to source CosyVoice2 model directory")
    parser.add_argument("--titw_test_dir", type=str, default="../../eval/titw-test",
                        help="Path to TITW test directory")
    parser.add_argument("--eval_dir", type=str, default="../../eval",
                        help="Path to eval directory containing generate_titw.py and eval_versa.py")
    parser.add_argument("--skip_epoch_eval", action="store_true",
                        help="Skip epoch evaluation (faster training without generation/eval)")

    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    embedding_dir = Path(args.embedding_dir)
    averaged = args.averaged
    
    # Resolve evaluation paths
    script_dir = Path(__file__).parent
    source_model_dir = Path(args.source_model_dir)
    if not source_model_dir.is_absolute():
        source_model_dir = (script_dir / source_model_dir).resolve()
    
    titw_test_dir = Path(args.titw_test_dir)
    if not titw_test_dir.is_absolute():
        titw_test_dir = (script_dir / titw_test_dir).resolve()
    
    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_absolute():
        eval_dir = (script_dir / eval_dir).resolve()

    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("=" * 80)
    print("Campplus Finetuning (Simple Pipeline)")
    print("=" * 80)
    
    # Load speaker embedding lookup (mean embeddings)
    print("\nLoading speaker embeddings...")
    if averaged:
        speaker_embedding_file = embedding_dir / "speaker_embedding_lookup.pt"
        if not speaker_embedding_file.exists():
            print(f"Error: {speaker_embedding_file} not found. Run generate_target_embeddings.py first.")
            sys.exit(1)
        speaker_embeddings = torch.load(speaker_embedding_file)
    else:
        speaker_embedding_file = embedding_dir / "target_embeddings.pt"
        if not speaker_embedding_file.exists():
            print(f"Error: {speaker_embedding_file} not found. Run generate_target_embeddings.py first.")
            sys.exit(1)
        speaker_embeddings = torch.load(speaker_embedding_file)
        speaker_embeddings = {
            spk_id: torch.tensor(data["embeddings"])
            for spk_id, data in speaker_embeddings.items()
        }


    print(f"Loaded embeddings for {len(speaker_embeddings)} speakers")
    
    # Load TITW metadata for epoch evaluation
    titw_metadata = None
    if not args.skip_epoch_eval:
        titw_metadata_dir = titw_test_dir / "metadata"
        if titw_metadata_dir.exists():
            print(f"\nLoading TITW test metadata from {titw_metadata_dir}...")
            titw_metadata = load_titw_metadata(titw_metadata_dir)
            print(f"Loaded metadata for {len(titw_metadata)} test utterances")
            print(f"Will evaluate with {args.eval_samples} random samples per epoch")
        else:
            print(f"Warning: TITW metadata not found at {titw_metadata_dir}, skipping epoch evaluation")
            args.skip_epoch_eval = True
    
    # Setup MLflow
    mlflow_config = config['mlflow']
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    mlflow.set_experiment(mlflow_config['experiment_name'])
    
    run_name = mlflow_config.get('run_name') or f"training_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log config
        mlflow.log_params({
            'epochs': config['training']['epochs'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'loss': 'cosine_similarity',
            'device': str(device),
            'num_workers': args.num_workers
        })
        
        # Create model
        print("\nCreating model...")
        campplus_path = config['campplus']['pretrained_path']
        
        # Resolve relative path
        if not os.path.isabs(campplus_path):
            campplus_path = str(Path(__file__).parent / campplus_path)
        
        model = create_model(campplus_path)
        model = model.to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Create optimizer
        training_config = config['training']
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        # Create dataset and dataloader
        print(f"\nInitializing dataset with {args.num_workers} workers...")
        if averaged:
            dataset = HFStreamingDataset(
                config=config,
                averaged_speaker_embeddings=speaker_embeddings,
                subset="easy"
            )
        else:
            dataset = HFStreamingDataset(
                config=config,
                speaker_embeddings=speaker_embeddings,
                subset="easy",
                averaged=False
            )
        
        dataloader = DataLoader(
            dataset,
            batch_size=training_config['batch_size'],
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True  # avoid batch norm issues with small last batch
        )
        
        # Create checkpoint directory
        checkpoint_dir = Path(training_config['checkpoint_dir'])
        if training_config.get("use_name_as_subdir", None):
            checkpoint_dir = checkpoint_dir / Path(args.config).stem
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        print("\nStarting training...")
        training_history = {
            'train_loss': [],
            'train_cosine_similarity': []
        }
        
        num_epochs = config['training']['epochs']
        best_loss = float('inf')
        best_path = None    
        prev_steps = 0  # To keep track of total steps for MLflow logging
        for epoch in range(num_epochs):
            # Train one epoch
            train_metrics = train_one_epoch(
                model=model,
                dataloader=dataloader,
                config=config,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                prev_steps=prev_steps
            )
            
            # Log metrics
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_cosine_similarity'].append(train_metrics['cosine_similarity'])
            prev_steps += train_metrics['num_batches']
            mlflow.log_metrics({
                'train_loss': train_metrics['loss'],
                'train_cosine_similarity': train_metrics['cosine_similarity'],
                'num_samples': train_metrics['num_samples']
            }, step=prev_steps)
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Cosine Sim: {train_metrics['cosine_similarity']:.4f}")
            print(f"  Samples processed: {train_metrics['num_samples']}")
            
            # Save checkpoint
            is_best = train_metrics['loss'] < best_loss
            if is_best:
                best_loss = train_metrics['loss']
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'training_history': training_history,
                'config': config
            }
            
            # Save latest checkpoint
            mlflow.pytorch.log_model(model, f"checkpoint_epoch_{epoch + 1}")
            print(f"  Saved checkpoint for epoch {epoch + 1} to MLflow")
            
            # Save best checkpoint
            if is_best:
                best_path = checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                print(f"  Saved best model to {best_path}")
            
            # Run epoch evaluation with random TITW samples
            if not args.skip_epoch_eval and titw_metadata is not None:
                eval_metrics = run_epoch_evaluation(
                    model=model,
                    config=config,
                    epoch=epoch,
                    checkpoint_dir=checkpoint_dir,
                    titw_metadata=titw_metadata,
                    num_eval_samples=args.eval_samples,
                    source_model_dir=str(source_model_dir),
                    titw_test_dir=titw_test_dir,
                    eval_dir=eval_dir
                )
                
                # Log evaluation metrics to MLflow
                if 'error' not in eval_metrics:
                    # Add eval metrics to training history
                    for key, value in eval_metrics.items():
                        if key != 'epoch' and isinstance(value, (int, float)):
                            if key not in training_history:
                                training_history[key] = []
                            training_history[key].append(value)
                    
                    mlflow.log_metrics(eval_metrics, step=prev_steps)
                    
                    # Print evaluation summary
                    print(f"\n  Epoch {epoch + 1} Evaluation Results:")
                    for key, value in eval_metrics.items():
                        if key != 'epoch' and isinstance(value, (int, float)):
                            print(f"    {key}: {value:.4f}")
                else:
                    print(f"  Epoch evaluation had error: {eval_metrics.get('error')}")


        # Log final model
        print("\nLogging model to MLflow...")
        mlflow.pytorch.log_model(model, "model")
        
        # also create ONNX model
        onnx_path = checkpoint_dir / "campplus_finetuned.onnx"
        print(f"\nExporting ONNX model to {onnx_path}...")
        export_onnx_model(model, str(onnx_path))
        mlflow.log_artifact(str(onnx_path))
        
        # Log training history
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        mlflow.log_artifact(str(history_path))
        
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best training loss: {best_loss:.4f}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()
