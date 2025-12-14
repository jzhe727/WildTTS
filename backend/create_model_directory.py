#!/usr/bin/env python3
"""
This script should create a directory similar to the one used in CosyVoice2
It should have symlinks to the large .pt and .onnx files and to any subfolders.
(excluiding the finetuned campplus.onnx, which should be copied)
It creates a sub-dirctory called CosyVoice2 inside the model_dir
Arguments:
    model_dir: Directory to create the model files in
    source_model_dir: Directory containing the source model files
    campplus_onnx_path: Path to the finetuned campplus.onnx file (default model_dir/campplus_finetuned.onnx)

Note: This is a copy of training/campplus_finetuning/create_model_directory.py
      for use in the backend deployment.
"""

import argparse
import os
import shutil
from pathlib import Path


def create_model_directory(model_dir: str, source_model_dir: str, campplus_onnx_path: str = None):
    """
    Create a model directory with symlinks to large files and copy of finetuned campplus.onnx
    
    Args:
        model_dir: Directory to create the model files in
        source_model_dir: Directory containing the source model files
        campplus_onnx_path: Path to the finetuned campplus.onnx file
    """
    model_dir = Path(model_dir)
    source_model_dir = Path(source_model_dir)
    
    # Default campplus path if not provided
    if campplus_onnx_path is None:
        campplus_onnx_path = model_dir / "campplus_finetuned.onnx"
    else:
        campplus_onnx_path = Path(campplus_onnx_path)
    
    # Create the CosyVoice2 subdirectory
    target_dir = model_dir / "CosyVoice2-0.5B"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating model directory at: {target_dir}")
    print(f"Source model directory: {source_model_dir}")
    print(f"Finetuned campplus path: {campplus_onnx_path}")
    
    # Iterate through all items in source directory
    for item in source_model_dir.iterdir():
        target_path = target_dir / item.name
        
        # Skip if target already exists
        if target_path.exists():
            print(f"Skipping {item.name} (already exists)")
            continue
        
        # Calculate relative path from target to source for symlinks
        # Use absolute path to ensure the symlink works correctly
        source_absolute = item.resolve()
        
        # Handle directories - create symlinks
        if item.is_dir():
            print(f"Creating symlink for directory: {item.name}")
            os.symlink(source_absolute, target_path, target_is_directory=True)
        
        # Handle files
        elif item.is_file():
            # Special handling for campplus.onnx - copy the finetuned version
            if item.name == "campplus.onnx":
                if campplus_onnx_path.exists():
                    print(f"Copying finetuned campplus.onnx to {target_path}")
                    shutil.copy2(campplus_onnx_path, target_path)
                else:
                    print(f"Warning: Finetuned campplus.onnx not found at {campplus_onnx_path}")
                    print("Creating symlink to original instead")
                    os.symlink(source_absolute, target_path)
            
            # For .pt and .onnx files, create symlinks
            elif item.suffix in ['.pt', '.onnx']:
                print(f"Creating symlink for: {item.name}")
                os.symlink(source_absolute, target_path)
            
            elif item.suffix in ['.txt', '.json', '.yaml', '.yml']:
                # Copy  small config files instead of symlinking
                print(f"Copying config file: {item.name}")
                shutil.copy2(item, target_path)
            # For other files, create symlinks as well
            else:
                print(f"Creating symlink for: {item.name}")
                os.symlink(source_absolute, target_path)
    
    print(f"\nModel directory created successfully at: {target_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a model directory with symlinks and finetuned campplus.onnx"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory to create the model files in"
    )
    parser.add_argument(
        "--source_model_dir",
        type=str,
        help="Directory containing the source model files"
    )
    parser.add_argument(
        "--campplus-onnx-path",
        type=str,
        default=None,
        help="Path to the finetuned campplus.onnx file (default: model_dir/campplus_finetuned.onnx)"
    )
    
    args = parser.parse_args()
    
    create_model_directory(
        args.model_dir,
        args.source_model_dir,
        args.campplus_onnx_path
    )


if __name__ == "__main__":
    main()
