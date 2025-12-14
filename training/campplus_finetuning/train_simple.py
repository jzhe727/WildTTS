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


def main():
    parser = argparse.ArgumentParser(description="Simple Campplus Finetuning")
    parser.add_argument("--config", type=str, default="averaged_config.yaml", help="Path to config file")
    parser.add_argument("--embedding_dir", type=str, default="./embedding_artifacts", help="Embedding artifacts directory")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--averaged", type=bool, default=True, help="Use averaged speaker embeddings or not")

    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    embedding_dir = Path(args.embedding_dir)
    averaged = args.averaged

    
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
        dataset = HFStreamingDataset(
            config=config,
            averaged_speaker_embeddings=speaker_embeddings,
            subset="easy"
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
