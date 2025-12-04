#!/usr/bin/env python3
"""
Campplus Finetuning Training Script

This script:
1. Loads prepared data and target embeddings
2. Finetunes the Campplus model
3. Logs training metrics to MLflow
4. Saves checkpoints and final model
5. Uploads model to HuggingFace Hub

Usage:
    python train.py --config config.yaml
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from dataset import CampplusDataset, create_dataloader, collate_fn
from model import CampplusModel, create_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


class Trainer:
    """Trainer class for Campplus finetuning."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        training_config = config['training']
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        # Learning rate scheduler
        if training_config['lr_scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['epochs'] - training_config['warmup_epochs']
            )
        else:
            self.scheduler = None
        
        # Warmup scheduler
        self.warmup_epochs = training_config['warmup_epochs']
        
        # Loss function
        if training_config['loss'] == 'cosine_embedding_loss':
            self.criterion = nn.CosineEmbeddingLoss(
                margin=training_config.get('margin', 0.0)
            )
        else:
            self.criterion = nn.MSELoss()
        
        # Gradient clipping
        self.gradient_clip = training_config.get('gradient_clip', 1.0)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=training_config.get('early_stopping_patience', 10)
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(training_config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = training_config.get('save_every_epochs', 5)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def _warmup_lr(self, epoch: int) -> float:
        """Compute warmup learning rate."""
        if epoch < self.warmup_epochs:
            return self.config['training']['learning_rate'] * (epoch + 1) / self.warmup_epochs
        return self.config['training']['learning_rate']
    
    def _update_lr(self, epoch: int):
        """Update learning rate with warmup."""
        if epoch < self.warmup_epochs:
            lr = self._warmup_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.scheduler is not None:
            self.scheduler.step()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch in pbar:
            features = batch['features'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            targets = batch['target_embeddings'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            embeddings = self.model(features, lengths)
            
            # Compute loss
            if isinstance(self.criterion, nn.CosineEmbeddingLoss):
                # For cosine embedding loss, we use target=1 (similar)
                labels = torch.ones(embeddings.shape[0], device=self.device)
                loss = self.criterion(embeddings, targets, labels)
            else:
                loss = self.criterion(embeddings, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_cosine_sim = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]"):
            features = batch['features'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            targets = batch['target_embeddings'].to(self.device)
            
            # Forward pass
            embeddings = self.model(features, lengths)
            
            # Compute loss
            if isinstance(self.criterion, nn.CosineEmbeddingLoss):
                labels = torch.ones(embeddings.shape[0], device=self.device)
                loss = self.criterion(embeddings, targets, labels)
            else:
                loss = self.criterion(embeddings, targets)
            
            # Compute cosine similarity
            cosine_sim = nn.functional.cosine_similarity(embeddings, targets).mean()
            
            total_loss += loss.item()
            total_cosine_sim += cosine_sim.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_cosine_sim = total_cosine_sim / num_batches
        
        return {
            'loss': avg_loss,
            'cosine_similarity': avg_cosine_sim
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def train(self, num_epochs: int) -> Dict[str, list]:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Update learning rate
            self._update_lr(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['learning_rate'].append(current_lr)
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_cosine_similarity': val_metrics['cosine_similarity'],
                'learning_rate': current_lr
            }, step=epoch)
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Cosine Sim: {val_metrics['cosine_similarity']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            # Save checkpoint
            if (epoch + 1) % self.save_every == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        return self.training_history


def upload_to_huggingface(
    model: nn.Module,
    config: dict,
    training_history: dict
):
    """Upload trained model to HuggingFace Hub."""
    hf_config = config['huggingface']
    
    if not hf_config.get('push_to_hub', False):
        print("HuggingFace upload disabled in config")
        return
    
    try:
        from huggingface_hub import HfApi, create_repo
        
        repo_id = hf_config['model_repo']
        
        # Create repo if it doesn't exist
        api = HfApi()
        try:
            create_repo(repo_id, private=hf_config.get('private', False))
        except Exception:
            pass  # Repo might already exist
        
        # Save model locally
        save_dir = Path("./hf_upload_temp")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
        
        # Save config
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save training history
        with open(save_dir / "training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Upload to Hub
        api.upload_folder(
            folder_path=str(save_dir),
            repo_id=repo_id,
            commit_message="Upload finetuned Campplus model"
        )
        
        print(f"Model uploaded to https://huggingface.co/{repo_id}")
        
    except ImportError:
        print("huggingface_hub not installed. Skipping HuggingFace upload.")
    except Exception as e:
        print(f"Error uploading to HuggingFace: {e}")


def main():
    parser = argparse.ArgumentParser(description="Finetune Campplus model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="./data_artifacts", help="Data artifacts directory")
    parser.add_argument("--embedding_dir", type=str, default="./embedding_artifacts", help="Embedding artifacts directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    data_dir = Path(args.data_dir)
    embedding_dir = Path(args.embedding_dir)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("=" * 80)
    print("Campplus Finetuning")
    print("=" * 80)
    
    # Setup MLflow
    mlflow_config = config['mlflow']
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    mlflow.set_experiment(mlflow_config['experiment_name'])
    
    run_name = mlflow_config.get('run_name') or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log config
        mlflow.log_params({
            'epochs': config['training']['epochs'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'loss': config['training']['loss'],
            'device': str(device)
        })
        
        # Create dataloaders
        print("\nCreating dataloaders...")
        
        target_embeddings_file = embedding_dir / "target_embeddings.pt"
        
        train_loader = create_dataloader(
            utt2spk_file=str(data_dir / "train_utt2spk"),
            wav_scp_file=str(data_dir / "train_wav.scp"),
            target_embeddings_file=str(target_embeddings_file),
            batch_size=config['training']['batch_size'],
            shuffle=True,
            data_root=str(config['data']['local_data_dir']),
            num_mel_bins=config['campplus']['num_mel_bins'],
            augment=config['training']['augmentation']['enabled']
        )
        
        val_loader = create_dataloader(
            utt2spk_file=str(data_dir / "val_utt2spk"),
            wav_scp_file=str(data_dir / "val_wav.scp"),
            target_embeddings_file=str(target_embeddings_file),
            batch_size=config['training']['batch_size'],
            shuffle=False,
            data_root=str(config['data']['local_data_dir']),
            num_mel_bins=config['campplus']['num_mel_bins'],
            augment=False
        )
        
        # Create model
        print("\nCreating model...")
        model = create_model(config['campplus'], pretrained=True)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"\nResuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.current_epoch = checkpoint['epoch'] + 1
            trainer.best_val_loss = checkpoint['best_val_loss']
            trainer.training_history = checkpoint['training_history']
        
        # Train
        print("\nStarting training...")
        training_history = trainer.train(config['training']['epochs'])
        
        # Log final model
        print("\nLogging model to MLflow...")
        mlflow.pytorch.log_model(model, "model")
        
        # Log training history
        history_path = trainer.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        mlflow.log_artifact(str(history_path))
        
        # Upload to HuggingFace
        print("\nUploading to HuggingFace...")
        upload_to_huggingface(model, config, training_history)
        
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()
