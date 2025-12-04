#!/usr/bin/env python3
"""
Evaluate CosyVoice2 with Finetuned Campplus Model

This script:
1. Loads the finetuned Campplus model
2. Generates TTS audio using CosyVoice2 with the finetuned speaker embeddings
3. Evaluates the generated audio using VERSA metrics
4. Logs all artifacts and metrics to MLflow

Usage:
    python evaluate.py --config config.yaml --checkpoint best_model.pt
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
import mlflow
from tqdm import tqdm

# Add parent directories to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / 'backend'))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / 'eval'))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_finetuned_model(checkpoint_path: str, config: dict, device: torch.device):
    """
    Load the finetuned Campplus model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    from model import create_model
    
    model = create_model(config['campplus'], pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded finetuned model from {checkpoint_path}")
    return model


def generate_speaker_embedding(
    model: torch.nn.Module,
    audio_path: str,
    config: dict,
    device: torch.device
) -> torch.Tensor:
    """
    Generate speaker embedding using the finetuned model.
    
    Args:
        model: Finetuned Campplus model
        audio_path: Path to audio file
        config: Configuration dictionary
        device: Device
        
    Returns:
        Speaker embedding tensor
    """
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    
    # Load audio
    speech, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    target_sr = config['data'].get('sample_rate', 16000)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        speech = resampler(speech)
    
    # Ensure mono
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    
    # Extract Fbank features
    feat = kaldi.fbank(
        speech,
        num_mel_bins=config['campplus'].get('num_mel_bins', 80),
        dither=0,
        sample_frequency=target_sr
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    
    # Generate embedding
    with torch.no_grad():
        feat = feat.unsqueeze(0).to(device)
        embedding = model(feat)
    
    return embedding.cpu()


def load_test_metadata(metadata_dir: Path) -> Dict[str, Dict]:
    """Load test metadata files."""
    metadata = {}
    
    # Load text_test
    text_dict = {}
    text_file = metadata_dir / 'text_test'
    if text_file.exists():
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    text_dict[utt_id] = text
    
    # Load wav_test.scp
    wav_dict = {}
    wav_file = metadata_dir / 'wav_test.scp'
    if wav_file.exists():
        with open(wav_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, wav_path = parts
                    wav_dict[utt_id] = wav_path
    
    # Load utt2spk_test
    spk_dict = {}
    spk_file = metadata_dir / 'utt2spk_test'
    if spk_file.exists():
        with open(spk_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    utt_id, spk_id = parts
                    spk_dict[utt_id] = spk_id
    
    # Combine
    for utt_id in text_dict.keys():
        if utt_id in wav_dict and utt_id in spk_dict:
            metadata[utt_id] = {
                'text': text_dict[utt_id],
                'wav_path': wav_dict[utt_id],
                'speaker_id': spk_dict[utt_id]
            }
    
    return metadata


def generate_tts_with_finetuned_embeddings(
    model: torch.nn.Module,
    test_metadata: Dict[str, Dict],
    test_wav_dir: Path,
    output_dir: Path,
    config: dict,
    device: torch.device
):
    """
    Generate TTS audio using CosyVoice2 with finetuned speaker embeddings.
    
    Args:
        model: Finetuned Campplus model
        test_metadata: Test metadata
        test_wav_dir: Directory containing test wav files
        output_dir: Output directory for generated audio
        config: Configuration dictionary
        device: Device
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Import CosyVoice2 TTS interface
    # This is a placeholder - adjust based on actual CosyVoice2 integration
    try:
        from tts import create_tts
        
        tts = create_tts(
            'cosyvoice',
            model_path=str(config['evaluation'].get('cosyvoice_model_path', '')),
            load_jit=False,
            load_trt=False,
            fp16=False
        )
        
        print(f"Generating TTS for {len(test_metadata)} utterances...")
        failed_utts = []
        
        for utt_id, utt_info in tqdm(test_metadata.items(), desc="Generating TTS"):
            try:
                prompt_wav_path = test_wav_dir / utt_info['wav_path']
                
                if not prompt_wav_path.exists():
                    failed_utts.append(utt_id)
                    continue
                
                # Generate speaker embedding using finetuned model
                speaker_embedding = generate_speaker_embedding(
                    model, str(prompt_wav_path), config, device
                )
                
                output_path = output_dir / f"{utt_id}.wav"
                
                # Synthesize with the embedding
                # Note: This may require modifying the TTS interface to accept
                # pre-computed embeddings
                tts.synthesize(
                    text=utt_info['text'],
                    prompt_wav_path=str(prompt_wav_path),
                    output_wav_path=str(output_path),
                    style_text=utt_info['text'],
                    speaker_embedding=speaker_embedding,  # Custom parameter
                    stream=False
                )
                
            except Exception as e:
                print(f"Error processing {utt_id}: {e}")
                failed_utts.append(utt_id)
                continue
        
        print(f"Generated: {len(test_metadata) - len(failed_utts)}, Failed: {len(failed_utts)}")
        
        if failed_utts:
            with open(output_dir / 'failed_utterances.txt', 'w') as f:
                for utt_id in failed_utts:
                    f.write(f"{utt_id}\n")
                    
    except ImportError as e:
        print(f"Warning: Could not import TTS module: {e}")
        print("TTS generation skipped. Please ensure CosyVoice2 is properly installed.")


def run_versa_evaluation(
    gt_dir: Path,
    pred_dir: Path,
    output_dir: Path,
    config_path: Path,
    text_file: Optional[Path] = None
) -> Dict:
    """
    Run VERSA evaluation on generated audio.
    
    Args:
        gt_dir: Ground truth audio directory
        pred_dir: Predicted audio directory
        output_dir: Output directory for results
        config_path: VERSA config file path
        text_file: Optional text file for WER
        
    Returns:
        Evaluation statistics
    """
    import subprocess
    
    # Create SCP files
    gt_scp = output_dir / "gt.scp"
    pred_scp = output_dir / "pred.scp"
    
    gt_files = sorted(gt_dir.rglob("*.wav"))
    pred_files = sorted(pred_dir.rglob("*.wav"))
    
    gt_dict = {f.stem: str(f.absolute()) for f in gt_files}
    pred_dict = {f.stem: str(f.absolute()) for f in pred_files}
    
    common_ids = set(gt_dict.keys()) & set(pred_dict.keys())
    
    with open(gt_scp, 'w') as gt_f, open(pred_scp, 'w') as pred_f:
        for utt_id in sorted(common_ids):
            gt_f.write(f"{utt_id} {gt_dict[utt_id]}\n")
            pred_f.write(f"{utt_id} {pred_dict[utt_id]}\n")
    
    # Run VERSA
    versa_script = Path(__file__).parent.parent.parent / "eval" / "versa" / "versa" / "bin" / "scorer.py"
    
    cmd = [
        sys.executable,
        str(versa_script),
        "--score_config", str(config_path),
        "--gt", str(gt_scp),
        "--pred", str(pred_scp),
        "--output_file", str(output_dir / "results.jsonl"),
        "--io", "soundfile"
    ]
    
    if text_file:
        cmd.extend(["--text", str(text_file)])
    
    if torch.cuda.is_available():
        cmd.extend(["--use_gpu", "true"])
    
    try:
        subprocess.run(cmd, check=True)
        print("VERSA evaluation completed")
    except subprocess.CalledProcessError as e:
        print(f"VERSA evaluation failed: {e}")
        return {}
    
    # Parse results
    results_file = output_dir / "results.jsonl"
    stats = {}
    
    if results_file.exists():
        results = {}
        with open(results_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                if 'key' in entry:
                    utt_id = entry.pop('key')
                    results[utt_id] = entry
        
        # Compute statistics
        metric_names = set()
        for utt_data in results.values():
            metric_names.update(utt_data.keys())
        
        for metric in sorted(metric_names):
            scores = []
            for utt_data in results.values():
                if metric in utt_data and isinstance(utt_data[metric], (int, float)):
                    scores.append(utt_data[metric])
            
            if scores:
                stats[metric] = {
                    'mean': sum(scores) / len(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
    
    return stats


def log_to_mlflow(
    config: dict,
    stats: Dict,
    output_dir: Path,
    checkpoint_path: str
):
    """Log evaluation results to MLflow."""
    mlflow_config = config['mlflow']
    
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    mlflow.set_experiment(mlflow_config['experiment_name'])
    
    run_name = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("checkpoint", checkpoint_path)
        mlflow.log_param("eval_config", config['evaluation']['cosyvoice_config'])
        
        # Log metrics
        for metric_name, metric_stats in stats.items():
            mlflow.log_metric(f"{metric_name}_mean", metric_stats['mean'])
            mlflow.log_metric(f"{metric_name}_min", metric_stats['min'])
            mlflow.log_metric(f"{metric_name}_max", metric_stats['max'])
        
        # Log artifacts
        mlflow.log_artifacts(str(output_dir), artifact_path="evaluation_results")
        
        print("Results logged to MLflow")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CosyVoice2 with finetuned Campplus")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./eval_output", help="Output directory")
    parser.add_argument("--skip_generation", action="store_true", help="Skip TTS generation")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("CosyVoice2 Evaluation with Finetuned Campplus")
    print("=" * 80)
    
    # Load finetuned model
    print("\nLoading finetuned model...")
    model = load_finetuned_model(args.checkpoint, config, device)
    
    # Load test metadata
    eval_config = config['evaluation']
    metadata_dir = Path(eval_config['test_metadata_dir'])
    test_wav_dir = Path(eval_config['test_wav_dir'])
    
    if not metadata_dir.is_absolute():
        metadata_dir = SCRIPT_DIR / metadata_dir
    if not test_wav_dir.is_absolute():
        test_wav_dir = SCRIPT_DIR / test_wav_dir
    
    print(f"\nLoading test metadata from {metadata_dir}...")
    test_metadata = load_test_metadata(metadata_dir)
    print(f"Loaded {len(test_metadata)} test utterances")
    
    # Generate TTS
    generated_dir = output_dir / "generated"
    
    if not args.skip_generation:
        print("\nGenerating TTS with finetuned embeddings...")
        generate_tts_with_finetuned_embeddings(
            model, test_metadata, test_wav_dir, generated_dir, config, device
        )
    else:
        print("\nSkipping TTS generation (--skip_generation)")
    
    # Run VERSA evaluation
    if generated_dir.exists() and any(generated_dir.glob("*.wav")):
        print("\nRunning VERSA evaluation...")
        
        versa_config = Path(eval_config['cosyvoice_config'])
        if not versa_config.is_absolute():
            versa_config = SCRIPT_DIR.parent.parent / "eval" / "configs" / versa_config.name
        
        stats = run_versa_evaluation(
            gt_dir=test_wav_dir,
            pred_dir=generated_dir,
            output_dir=output_dir / "versa_results",
            config_path=versa_config
        )
        
        # Print results
        print("\nEvaluation Results:")
        print("-" * 40)
        for metric, values in stats.items():
            print(f"{metric}: {values['mean']:.4f} (±{values['max']-values['min']:.4f})")
        
        # Log to MLflow
        print("\nLogging to MLflow...")
        log_to_mlflow(config, stats, output_dir, args.checkpoint)
    else:
        print("\nNo generated audio found for evaluation")
    
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
