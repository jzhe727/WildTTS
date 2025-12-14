#!/usr/bin/env python3
"""
VERSA Evaluation Pipeline for TTS Generated Audio

This script evaluates generated audio files using the VERSA toolkit.
Currently supports UTMOS metric, but designed to be extensible for additional metrics.

Usage:
    python eval_versa.py --config configs/utmos.yaml --output_dir results/utmos
    python eval_versa.py --config configs/custom.yaml --gt_dir custom_gt --pred_dir custom_pred
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


def setup_logging(output_dir: Path, verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_file = output_dir / "evaluation.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_scp_file(scp_path: Path) -> Dict[str, str]:
    """
    Load Kaldi-style SCP file.
    
    Args:
        scp_path: Path to SCP file
        
    Returns:
        Dictionary mapping utterance IDs to file paths
    """
    scp_dict = {}
    with open(scp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, filepath = parts
                    scp_dict[utt_id] = filepath
    return scp_dict


def create_scp_files(gt_dir: Path, pred_dir: Path, output_dir: Path, logger) -> tuple:
    """
    Create SCP files for ground truth and predicted audio files.
    
    Args:
        gt_dir: Directory containing ground truth audio files
        pred_dir: Directory containing predicted/generated audio files
        output_dir: Directory to save SCP files
        logger: Logger instance
        
    Returns:
        Tuple of (gt_scp_path, pred_scp_path)
    """
    gt_scp_path = output_dir / "gt.scp"
    pred_scp_path = output_dir / "pred.scp"
    
    # Get all wav files from ground truth directory
    gt_files = sorted(gt_dir.rglob("*.wav"))
    pred_files = sorted(pred_dir.rglob("*.wav"))
    
    # Create mapping from filename to full path
    gt_dict = {f.stem: str(f.absolute()) for f in gt_files}
    pred_dict = {f.stem: str(f.absolute()) for f in pred_files}
    
    # Find common utterance IDs
    common_ids = set(gt_dict.keys()) & set(pred_dict.keys())
    
    if not common_ids:
        logger.error("No common utterance IDs found between GT and predicted files!")
        logger.info(f"GT files: {len(gt_dict)}, Pred files: {len(pred_dict)}")
        raise ValueError("No common files to evaluate")
    
    logger.info(f"Found {len(common_ids)} common utterances to evaluate")
    logger.info(f"GT total: {len(gt_dict)}, Pred total: {len(pred_dict)}")
    
    # Write SCP files
    with open(gt_scp_path, 'w') as gt_f, open(pred_scp_path, 'w') as pred_f:
        for utt_id in sorted(common_ids):
            gt_f.write(f"{utt_id} {gt_dict[utt_id]}\n")
            pred_f.write(f"{utt_id} {pred_dict[utt_id]}\n")
    
    logger.info(f"Created SCP files: {gt_scp_path}, {pred_scp_path}")
    return gt_scp_path, pred_scp_path


def load_versa_results(result_file: Path, logger) -> Dict:
    """
    Load and parse VERSA results JSONL file.
    
    Args:
        result_file: Path to VERSA results JSONL file
        logger: Logger instance
        
    Returns:
        Dictionary mapping utterance IDs to their metrics
    """
    if not result_file.exists():
        logger.error(f"Results file not found: {result_file}")
        return {}
    
    results = {}
    incomplete_entries = []
    with open(result_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Each line should have a 'key' field as the utterance ID
                if 'key' in entry:
                    utt_id = entry.pop('key')
                    # Check if entry has any metrics (more than just the key)
                    if not entry:
                        incomplete_entries.append(utt_id)
                    results[utt_id] = entry
                else:
                    logger.warning(f"Line {line_num}: No 'key' field found, skipping")
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Failed to parse JSON: {e}")
                continue
    
    logger.info(f"Loaded {len(results)} utterance results from {result_file}")
    
    # Warn about incomplete entries
    if incomplete_entries:
        logger.warning(f"Found {len(incomplete_entries)} utterances with missing metrics:")
        for utt_id in incomplete_entries:
            logger.warning(f"  - {utt_id}")
    
    return results


def compute_statistics(results: Dict, logger) -> Dict:
    """
    Compute statistics from VERSA results.
    
    Args:
        results: Dictionary mapping utterance IDs to metric dictionaries
                 Structure: {utt_id: {metric_name: score, ...}, ...}
        logger: Logger instance
        
    Returns:
        Dictionary with computed statistics for each metric
    """
    stats = {}
    
    if not results:
        logger.warning("No results to compute statistics")
        return stats
    
    logger.info(f"Computing statistics for {len(results)} utterances")
    
    # Collect all metric names from all utterances
    metric_names = set()
    for utt_data in results.values():
        if isinstance(utt_data, dict):
            metric_names.update(utt_data.keys())
    
    logger.info(f"Found metrics: {', '.join(sorted(metric_names))}")
    
    # Compute statistics for each metric
    for metric in sorted(metric_names):
        scores = []
        for utt_id, utt_data in results.items():
            if isinstance(utt_data, dict) and metric in utt_data:
                score = utt_data[metric]
                # Only compute statistics for numeric values
                if isinstance(score, (int, float)):
                    scores.append(score)
        
        if scores:
            stats[metric] = {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'count': len(scores)
            }
            logger.info(
                f"{metric}: mean={stats[metric]['mean']:.4f}, "
                f"min={stats[metric]['min']:.4f}, "
                f"max={stats[metric]['max']:.4f}, "
                f"n={stats[metric]['count']}"
            )
        else:
            logger.warning(f"{metric}: No numeric scores found")
    
    # Calculate WER from word stats

    """
    "whisper_wer_delete": {
    "mean": 0.33897932248130225,
    "min": 0,
    "max": 18,
    "count": 9092
  },
  "whisper_wer_equal": {
    "mean": 9.79762428508579,
    "min": 0,
    "max": 45,
    "count": 9092
  },
  "whisper_wer_insert": {
    "mean": 0.18554773427188737,
    "min": 0,
    "max": 15,
    "count": 9092
  },
  "whisper_wer_replace": {
    "mean": 0.4965904091509019,
    "min": 0,
    "max": 37,
    "count": 9092
  }
    """
    wer_stats = ["whisper_wer_delete", "whisper_wer_equal", "whisper_wer_insert", "whisper_wer_replace"]
    if all(stat in stats for stat in wer_stats):
        original_length = stats["whisper_wer_equal"]["mean"] + stats["whisper_wer_delete"]["mean"] + stats["whisper_wer_replace"]["mean"]
        whisper_wer_percentage = (stats["whisper_wer_delete"]["mean"] + stats["whisper_wer_insert"]["mean"] + stats["whisper_wer_replace"]["mean"]) / original_length * 100
        stats["whisper_wer_percentage"] = {
            
            "mean": whisper_wer_percentage,
            "min": 0.0,
            "max": 0.0,
            "count": 1,
        }

    return stats


def run_versa_evaluation(
    versa_script: Path, config_path: Path,
    gt_scp: Path, pred_scp: Path,
    output_file: Path, logger, text_file: Optional[Path] = None
) -> int:
    """
    Run VERSA evaluation using the scorer script.
    
    Args:
        versa_script: Path to VERSA scorer.py
        config_path: Path to VERSA config YAML
        gt_scp: Path to ground truth SCP file
        pred_scp: Path to predicted SCP file
        output_file: Path to save results (without extension)
        logger: Logger instance
        text_file: Optional path to text transcription file for WER
        
    Returns:
        Exit code (0 for success)
    """
    import subprocess
    
    cmd = [
        sys.executable,
        str(versa_script),
        "--score_config", str(config_path),
        "--gt", str(gt_scp),
        "--pred", str(pred_scp),
        "--output_file", str(output_file),
        "--io", "soundfile"
    ]
    
    # Add text file if provided (needed for WER calculation)
    if text_file is not None:
        cmd.extend(["--text", str(text_file)])

    # Add --use_gpu if gpu is available
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        cmd.extend(["--use_gpu", "true"])
    
    logger.info(f"Running VERSA evaluation: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("VERSA evaluation completed successfully")
        if result.stdout:
            logger.debug(f"STDOUT: {result.stdout}")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"VERSA evaluation failed with exit code {e.returncode}")
        logger.error(f"STDERR: {e.stderr}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        return e.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TTS generated audio using VERSA toolkit"
    )
    
    # Input/Output paths
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=Path("titw-test"),
        help="Directory containing ground truth audio files (default: titw-test)"
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        default=Path("titw_generated"),
        help="Directory containing predicted/generated audio files (default: titw_generated)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("eval_results"),
        help="Directory to save evaluation results (default: eval_results)"
    )
    
    # VERSA configuration
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/utmos.yaml"),
        help="Path to VERSA config file (default: configs/utmos.yaml)"
    )
    parser.add_argument(
        "--versa_dir",
        type=Path,
        default=Path("versa"),
        help="Path to VERSA toolkit directory (default: versa)"
    )
    
    # Optional inputs
    parser.add_argument(
        "--gt_scp",
        type=Path,
        help="Pre-existing ground truth SCP file (optional)"
    )
    parser.add_argument(
        "--pred_scp",
        type=Path,
        help="Pre-existing predicted SCP file (optional)"
    )
    parser.add_argument(
        "--text",
        type=Path,
        help="Text transcription file for WER calculation (Kaldi format: utt_id text)"
    )
    
    # Options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--skip_versa",
        action="store_true",
        help="Skip VERSA evaluation and only compute statistics from existing results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.verbose)
    logger.info("=" * 80)
    logger.info("VERSA Evaluation Pipeline")
    logger.info("=" * 80)
    logger.info(f"Ground truth directory: {args.gt_dir}")
    logger.info(f"Predicted directory: {args.pred_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Config file: {args.config}")
    
    # Validate paths
    if not args.gt_dir.exists():
        logger.error(f"Ground truth directory not found: {args.gt_dir}")
        return 1
    
    if not args.pred_dir.exists():
        logger.error(f"Predicted directory not found: {args.pred_dir}")
        return 1
    
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    versa_script = args.versa_dir / "versa" / "bin" / "scorer.py"
    if not versa_script.exists():
        logger.error(f"VERSA scorer script not found: {versa_script}")
        logger.error(f"Please check --versa_dir argument (current: {args.versa_dir})")
        return 1
    
    # Create or use existing SCP files
    if args.gt_scp and args.pred_scp:
        logger.info("Using pre-existing SCP files")
        gt_scp = args.gt_scp
        pred_scp = args.pred_scp
    else:
        if not args.skip_versa:
            logger.info("Creating SCP files from directories")
            try:
                gt_scp, pred_scp = create_scp_files(
                    args.gt_dir, args.pred_dir, args.output_dir, logger
                )
            except Exception as e:
                logger.error(f"Failed to create SCP files: {e}")
                return 1
    
    # Run VERSA evaluation
    result_file = args.output_dir / "results.jsonl"
    
    if not args.skip_versa:
        exit_code = run_versa_evaluation(
            versa_script, args.config, gt_scp, pred_scp,
            args.output_dir / "results.jsonl", logger, args.text
        )
        
        if exit_code != 0:
            logger.error("VERSA evaluation failed")
            return exit_code
    else:
        logger.info("Skipping VERSA evaluation (--skip_versa flag set)")
    
    # Load and analyze results
    if result_file.exists():
        logger.info("Loading evaluation results")
        results = load_versa_results(result_file, logger)
        
        logger.info("Computing statistics")
        stats = compute_statistics(results, logger)
        
        # Save statistics
        stats_file = args.output_dir / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {stats_file}")
        
        # Create summary report
        summary_file = args.output_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VERSA Evaluation Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Ground Truth: {args.gt_dir}\n")
            f.write(f"Predicted: {args.pred_dir}\n")
            f.write(f"Config: {args.config}\n\n")
            f.write("Metrics:\n")
            f.write("-" * 80 + "\n")
            for metric, values in stats.items():
                f.write(f"\n{metric}:\n")
                f.write(f"  Mean:  {values['mean']:.4f}\n")
                f.write(f"  Min:   {values['min']:.4f}\n")
                f.write(f"  Max:   {values['max']:.4f}\n")
                f.write(f"  Count: {values['count']}\n")
        logger.info(f"Summary saved to {summary_file}")
    else:
        logger.warning(f"Results file not found: {result_file}")
        if not args.skip_versa:
            logger.error("Evaluation may have failed")
            return 1
    
    logger.info("=" * 80)
    logger.info("Evaluation pipeline completed successfully")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
