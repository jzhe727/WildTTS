#!/usr/bin/env python3
"""
Campplus Model for Finetuning

This module provides a PyTorch wrapper around the Campplus ONNX model
that enables gradient-based finetuning.

Uses onnx2torch to automatically convert the ONNX model to PyTorch,
preserving the exact architecture and weights while enabling gradient flow.
"""

import torch
import torch.nn as nn
from onnx2torch import convert


class CampplusFinetunableModel(nn.Module):
    """
    Campplus model converted from ONNX for finetuning.
    Uses onnx2torch for automatic conversion with full gradient support.
    
    Note: Model must be in eval() mode when using batch_size=1 due to BatchNorm layers.
    For training, use batch_size > 1 or freeze BatchNorm layers.
    """
    
    def __init__(self, onnx_path: str):
        super().__init__()
        
        # Convert ONNX to PyTorch using onnx2torch
        self.backbone = convert(onnx_path)
        
        # Get embedding dimension by doing a forward pass
        with torch.no_grad():
            dummy_input = torch.randn(2, 100, 80)  # (batch, time, features) - use batch_size=2
            self.backbone.eval()
            dummy_output = self.backbone(dummy_input)
            self.embedding_dim = dummy_output.shape[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from input features.
        
        Args:
            x: Input Fbank features of shape (batch, time, features)
            
        Returns:
            Speaker embeddings of shape (batch, embedding_dim)
        """
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all parameters for feature extraction only."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters for full finetuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_batchnorm(self):
        """
        Freeze BatchNorm layers to allow batch_size=1 training.
        This keeps the running statistics from pretrained model.
        """
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False


def create_model(onnx_path: str) -> CampplusFinetunableModel:
    """
    Create Campplus model from ONNX file.
    
    Args:
        onnx_path: Path to campplus ONNX model
        
    Returns:
        Converted PyTorch model ready for finetuning
    """
    model = CampplusFinetunableModel(onnx_path)
    return model
