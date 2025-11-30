#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate TTS audio files for TITW test set using CosyVoice2 zero-shot inference.

This script uses the pretrained CosyVoice2-0.5B model to generate audio based on
each utterance as its own prompt (TITW reconstruction task). Each test utterance
uses its own wav file as the prompt audio and its own text as both the prompt
transcript and generation target. It processes 9113 test utterances from the 
TITW-test dataset.

Format compatibility with Versa toolkit:
- Generates WAV files with proper naming convention
- Maintains speaker identity through zero-shot voice cloning
- Outputs can be used directly with Versa evaluation toolkit
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add backend to path for TTS interface
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent / 'backend'
sys.path.insert(0, str(BACKEND_DIR))

from tts import create_tts


def load_metadata(metadata_dir):
    """
    Load test metadata files.
    
    Args:
        metadata_dir: Path to metadata directory
        
    Returns:
        dict: Mapping of utterance_id to {text, wav_path, speaker_id}
    """
    metadata_dir = Path(metadata_dir)
    
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


def generate_tts_audio(args):
    """
    Main function to generate TTS audio for all test utterances.
    
    Args:
        args: Command line arguments
    """
    # Set up paths
    metadata_dir = Path(args.metadata_dir)
    test_wav_dir = Path(args.test_wav_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("Loading metadata...")
    metadata = load_metadata(metadata_dir)
    print(f"Loaded {len(metadata)} test utterances")
    
    print("Note: Using each utterance as its own prompt for TITW reconstruction")
    
    # Initialize TTS engine using the unified interface
    print(f"Loading CosyVoice2 model from {args.model_dir}...")
    tts = create_tts(
        'cosyvoice',
        model_path=str(args.model_dir),
        load_jit=args.load_jit,
        load_trt=args.load_trt,
        fp16=args.fp16
    )
    print("Model loaded successfully")
    
    # Generate audio for each utterance
    print("Generating TTS audio...")
    failed_utts = []
    
    for utt_id, utt_info in tqdm(metadata.items(), desc="Processing utterances"):
        try:
            # Use the utterance itself as the prompt (TITW reconstruction)
            # Both the prompt wav and text come from the same utterance
            prompt_wav_path = test_wav_dir / utt_info['wav_path']
            prompt_text = utt_info['text']
            
            # Target text to synthesize (same as prompt text for TITW)
            tts_text = utt_info['text']
            
            # Check if prompt wav file exists
            if not prompt_wav_path.exists():
                print(f"Warning: Prompt wav file not found: {prompt_wav_path}, skipping {utt_id}")
                failed_utts.append(utt_id)
                continue
            
            # Output path for generated audio
            output_path = output_dir / f"{utt_id}.wav"
            
            # Generate audio using the TTS interface
            tts.synthesize(
                text=tts_text,
                prompt_wav_path=str(prompt_wav_path),
                output_wav_path=str(output_path),
                style_text=prompt_text,
                stream=False
            )
            
        except Exception as e:
            print(f"Error processing {utt_id}: {str(e)}")
            failed_utts.append(utt_id)
            continue
    
    # Summary
    print("\nGeneration complete!")
    print(f"Successfully generated: {len(metadata) - len(failed_utts)} files")
    print(f"Failed: {len(failed_utts)} files")
    
    if failed_utts:
        failed_list_path = output_dir / 'failed_utterances.txt'
        with open(failed_list_path, 'w') as f:
            for utt_id in failed_utts:
                f.write(f"{utt_id}\n")
        print(f"Failed utterances saved to: {failed_list_path}")
    
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate TTS audio for TITW test set using CosyVoice2'
    )
    
    # Paths
    parser.add_argument(
        '--metadata_dir',
        type=str,
        default='titw-test/metadata',
        help='Path to metadata directory containing text_test, wav_test.scp, etc.'
    )
    parser.add_argument(
        '--test_wav_dir',
        type=str,
        default='titw-test',
        help='Directory containing test wav files (parent of test/ folder)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='titw_generated',
        help='Output directory for generated audio files'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='../backend/CosyVoice/pretrained_models/CosyVoice2-0.5B',
        help='Path to CosyVoice2 model directory'
    )
    
    # Model options
    parser.add_argument(
        '--load_jit',
        action='store_true',
        help='Load JIT optimized model'
    )
    parser.add_argument(
        '--load_trt',
        action='store_true',
        help='Load TensorRT optimized model'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use FP16 precision'
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).resolve().parent
    args.metadata_dir = (script_dir / args.metadata_dir).resolve()
    args.test_wav_dir = (script_dir / args.test_wav_dir).resolve()
    args.output_dir = (script_dir / args.output_dir).resolve()
    args.model_dir = (script_dir / args.model_dir).resolve()
    
    print("="*80)
    print("TTS Generation for TITW Test Set")
    print("="*80)
    print(f"Metadata directory: {args.metadata_dir}")
    print(f"Test wav directory: {args.test_wav_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Load JIT: {args.load_jit}")
    print(f"Load TRT: {args.load_trt}")
    print(f"FP16: {args.fp16}")
    print("="*80)
    
    generate_tts_audio(args)


if __name__ == '__main__':
    main()
