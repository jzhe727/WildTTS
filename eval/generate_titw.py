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

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio
import librosa

# Add CosyVoice to path
SCRIPT_DIR = Path(__file__).resolve().parent
COSYVOICE_DIR = SCRIPT_DIR.parent / 'backend' / 'CosyVoice'
sys.path.insert(0, str(COSYVOICE_DIR))
sys.path.insert(0, str(COSYVOICE_DIR / 'third_party' / 'Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav


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


def postprocess_audio(speech, top_db=60, hop_length=220, win_length=440, max_val=0.8):
    """
    Post-process generated audio: trim silence and normalize.
    
    Args:
        speech: torch.Tensor audio waveform
        top_db: Decibel threshold for silence trimming
        hop_length: Hop length for trimming
        win_length: Window length for trimming
        max_val: Maximum amplitude value for normalization
        
    Returns:
        torch.Tensor: Processed audio
    """
    speech, _ = librosa.effects.trim(
        speech.numpy(),
        top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    speech = torch.from_numpy(speech)
    
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    
    return speech


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
    
    # Initialize CosyVoice2 model
    print(f"Loading CosyVoice2 model from {args.model_dir}...")
    cosyvoice = CosyVoice2(
        str(args.model_dir),
        load_jit=args.load_jit,
        load_trt=args.load_trt,
        fp16=args.fp16
    )
    print("Model loaded successfully")
    
    # Set sample rate for prompt audio
    prompt_sr = 16000
    
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
            
            # Load and preprocess prompt audio
            prompt_speech_16k = load_wav(str(prompt_wav_path), prompt_sr)
            
            # Optionally postprocess prompt audio
            if args.postprocess_prompt:
                prompt_speech_16k = postprocess_audio(prompt_speech_16k)
            
            # Generate audio using zero-shot inference
            output_audio = None
            for output in cosyvoice.inference_zero_shot(
                tts_text=tts_text,
                prompt_text=prompt_text,
                prompt_speech_16k=prompt_speech_16k,
                stream=False,
                speed=args.speed
            ):
                output_audio = output['tts_speech']
            
            if output_audio is None:
                print(f"Warning: Failed to generate audio for {utt_id}")
                failed_utts.append(utt_id)
                continue
            
            # Postprocess generated audio
            if args.postprocess_output:
                output_audio = postprocess_audio(output_audio.squeeze(0))
                output_audio = output_audio.unsqueeze(0)
            
            # Save audio file
            output_path = output_dir / f"{utt_id}.wav"
            torchaudio.save(
                str(output_path),
                output_audio.cpu(),
                cosyvoice.sample_rate
            )
            
        except Exception as e:
            print(f"Error processing {utt_id}: {str(e)}")
            failed_utts.append(utt_id)
            continue
    
    # Summary
    print(f"\nGeneration complete!")
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
    
    # Generation options
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Speed factor for generation'
    )
    parser.add_argument(
        '--postprocess_prompt',
        action='store_true',
        help='Apply post-processing to prompt audio'
    )
    parser.add_argument(
        '--postprocess_output',
        action='store_true',
        help='Apply post-processing to generated audio'
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
    print(f"Speed: {args.speed}")
    print(f"Postprocess prompt: {args.postprocess_prompt}")
    print(f"Postprocess output: {args.postprocess_output}")
    print("="*80)
    
    generate_tts_audio(args)


if __name__ == '__main__':
    main()
