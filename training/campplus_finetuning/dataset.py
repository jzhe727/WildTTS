#!/usr/bin/env python3
"""
Campplus Finetuning Dataset and DataLoader

This module provides PyTorch Dataset classes for loading audio samples
and their corresponding target speaker embeddings.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CampplusDataset(Dataset):
    """
    Dataset for Campplus finetuning.
    
    Each sample consists of:
    - Audio features (Fbank)
    - Target speaker embedding (randomly selected from top samples)
    """
    
    def __init__(
        self,
        utt2spk_file: str,
        wav_scp_file: str,
        target_embeddings_file: str,
        data_root: str = "",
        num_mel_bins: int = 80,
        sample_rate: int = 16000,
        max_frames: Optional[int] = None,
        augment: bool = False,
        noise_snr_range: Tuple[float, float] = (10, 30),
        speed_perturb: Tuple[float, float] = (0.9, 1.1)
    ):
        """
        Initialize the dataset.
        
        Args:
            utt2spk_file: Path to utt2spk file
            wav_scp_file: Path to wav.scp file
            target_embeddings_file: Path to target_embeddings.pt
            data_root: Root directory for resolving relative paths
            num_mel_bins: Number of mel bins for Fbank
            sample_rate: Target sample rate
            max_frames: Maximum number of frames (truncate longer samples)
            augment: Whether to apply data augmentation
            noise_snr_range: SNR range for noise augmentation
            speed_perturb: Speed perturbation range
        """
        self.data_root = Path(data_root)
        self.num_mel_bins = num_mel_bins
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.augment = augment
        self.noise_snr_range = noise_snr_range
        self.speed_perturb = speed_perturb
        
        # Load utt2spk
        self.utt2spk = {}
        with open(utt2spk_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    utt_id, spk_id = parts
                    self.utt2spk[utt_id] = spk_id
        
        # Load wav.scp
        self.wav_scp = {}
        with open(wav_scp_file, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, wav_path = parts
                    self.wav_scp[utt_id] = wav_path
        
        # Load target embeddings
        self.target_embeddings = torch.load(target_embeddings_file)
        
        # Build sample list (only include utterances with valid speakers)
        self.samples = []
        for utt_id, spk_id in self.utt2spk.items():
            if utt_id in self.wav_scp and spk_id in self.target_embeddings:
                self.samples.append({
                    "utt_id": utt_id,
                    "spk_id": spk_id,
                    "wav_path": self.wav_scp[utt_id]
                })
        
        print(f"Loaded {len(self.samples)} samples from {len(set(self.utt2spk.values()))} speakers")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _extract_features(self, wav_path: str) -> torch.Tensor:
        """Extract Fbank features from audio file."""
        # Resolve path
        if not os.path.isabs(wav_path):
            wav_path = str(self.data_root / wav_path)
        
        # Load audio
        speech, sr = torchaudio.load(wav_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            speech = resampler(speech)
        
        # Ensure mono
        if speech.shape[0] > 1:
            speech = speech.mean(dim=0, keepdim=True)
        
        # Apply augmentation if enabled
        if self.augment:
            speech = self._apply_augmentation(speech)
        
        # Extract Fbank features
        feat = kaldi.fbank(
            speech,
            num_mel_bins=self.num_mel_bins,
            dither=0,
            sample_frequency=self.sample_rate
        )
        
        # Mean normalization
        feat = feat - feat.mean(dim=0, keepdim=True)
        
        # Truncate if needed
        if self.max_frames is not None and feat.shape[0] > self.max_frames:
            # Random crop
            start = random.randint(0, feat.shape[0] - self.max_frames)
            feat = feat[start:start + self.max_frames]
        
        return feat
    
    def _apply_augmentation(self, speech: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to speech."""
        # Speed perturbation
        if random.random() < 0.5:
            speed = random.uniform(*self.speed_perturb)
            effects = [["speed", str(speed)], ["rate", str(self.sample_rate)]]
            speech, _ = torchaudio.sox_effects.apply_effects_tensor(
                speech, self.sample_rate, effects
            )
        
        # Add noise (simple white noise)
        if random.random() < 0.3:
            snr = random.uniform(*self.noise_snr_range)
            noise = torch.randn_like(speech) * 0.01
            speech_power = speech.pow(2).mean()
            noise_power = noise.pow(2).mean()
            scale = (speech_power / (noise_power * (10 ** (snr / 10)))).sqrt()
            speech = speech + noise * scale
        
        return speech
    
    def _get_target_embedding(self, spk_id: str) -> torch.Tensor:
        """
        Get target embedding for a speaker.
        Randomly selects from the top samples embeddings.
        """
        spk_data = self.target_embeddings[spk_id]
        embeddings = spk_data["embeddings"]
        
        # Randomly select one embedding
        idx = random.randint(0, len(embeddings) - 1)
        return embeddings[idx]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Extract features
        features = self._extract_features(sample["wav_path"])
        
        # Get target embedding
        target_embedding = self._get_target_embedding(sample["spk_id"])
        
        return {
            "features": features,
            "target_embedding": target_embedding,
            "utt_id": sample["utt_id"],
            "spk_id": sample["spk_id"]
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    Handles variable length sequences with padding.
    """
    # Get max length
    max_len = max(item["features"].shape[0] for item in batch)
    
    # Pad features
    batch_size = len(batch)
    num_mel_bins = batch[0]["features"].shape[1]
    
    padded_features = torch.zeros(batch_size, max_len, num_mel_bins)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    target_embeddings = []
    utt_ids = []
    spk_ids = []
    
    for i, item in enumerate(batch):
        feat_len = item["features"].shape[0]
        padded_features[i, :feat_len] = item["features"]
        lengths[i] = feat_len
        target_embeddings.append(item["target_embedding"])
        utt_ids.append(item["utt_id"])
        spk_ids.append(item["spk_id"])
    
    return {
        "features": padded_features,
        "lengths": lengths,
        "target_embeddings": torch.stack(target_embeddings),
        "utt_ids": utt_ids,
        "spk_ids": spk_ids
    }


def create_dataloader(
    utt2spk_file: str,
    wav_scp_file: str,
    target_embeddings_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for training.
    
    Args:
        utt2spk_file: Path to utt2spk file
        wav_scp_file: Path to wav.scp file
        target_embeddings_file: Path to target_embeddings.pt
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional arguments for CampplusDataset
        
    Returns:
        DataLoader instance
    """
    dataset = CampplusDataset(
        utt2spk_file=utt2spk_file,
        wav_scp_file=wav_scp_file,
        target_embeddings_file=target_embeddings_file,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
