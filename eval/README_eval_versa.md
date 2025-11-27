# VERSA Evaluation Pipeline

A comprehensive evaluation pipeline for TTS-generated audio using the VERSA (Versatile Evaluation of Speech and Audio) toolkit.

## Overview

This pipeline evaluates generated audio files against ground truth using objective metrics. Currently supports UTMOS (Universal Test MOS) with an extensible architecture for adding more metrics.

## Features

- **UTMOS Evaluation**: Predicts Mean Opinion Score (MOS) for audio naturalness (1-5 scale)
- **Extensible Architecture**: Easy to add new metrics via configuration files
- **Flexible Input**: Supports directory-based or SCP file inputs
- **Comprehensive Reporting**: JSON results, statistics, and text summaries
- **Logging**: Detailed logging for debugging and tracking
- **Shell Script Interface**: Simple command-line interface for common use cases

## Quick Start

### 1. Basic UTMOS Evaluation

```bash
cd /home/john.zheng1/ensf619/WildTTS/eval
chmod +x eval_versa.sh
./eval_versa.sh
```

This will evaluate audio in `titw_generated/` against ground truth in `titw-test/` using UTMOS.

### 2. Comprehensive Evaluation (Multiple Metrics)

```bash
./eval_versa.sh comprehensive
```

This runs UTMOS, DNSMOS, and PLCMOS metrics.

### 3. Custom Evaluation

```bash
./eval_versa.sh custom configs/my_config.yaml results/my_output
```

## Installation Requirements

### Required
- Python 3.7+
- VERSA toolkit (already present in `versa/`)

### Python Dependencies
```bash
# Core dependencies (likely already installed)
pip install numpy scipy soundfile

# UTMOS dependencies (if not auto-installed)
cd versa
python tools/install_utmos.py  # If needed
```

## Usage

### Python Script

```bash
python eval_versa.py \
    --gt_dir titw-test \
    --pred_dir titw_generated \
    --output_dir eval_results \
    --config configs/utmos.yaml \
    --versa_dir versa
```

**Arguments:**
- `--gt_dir`: Directory containing ground truth audio files
- `--pred_dir`: Directory containing predicted/generated audio files
- `--output_dir`: Directory to save evaluation results
- `--config`: Path to VERSA configuration file
- `--versa_dir`: Path to VERSA toolkit directory
- `--gt_scp`: (Optional) Pre-existing ground truth SCP file
- `--pred_scp`: (Optional) Pre-existing predicted SCP file
- `--verbose`: Enable verbose logging
- `--skip_versa`: Skip evaluation, only compute statistics from existing results

### Shell Script

```bash
# Basic usage
./eval_versa.sh

# Comprehensive evaluation
./eval_versa.sh comprehensive

# Custom configuration
./eval_versa.sh custom <config_file> <output_dir>

# Verbose mode
./eval_versa.sh -v

# Help
./eval_versa.sh --help

# Custom directories via environment variables
GT_DIR=custom_gt PRED_DIR=custom_pred ./eval_versa.sh
```

## Configuration Files

Configuration files define which metrics to compute. They are located in `configs/`.

### configs/utmos.yaml (Default)
```yaml
- name: pseudo_mos
  predictor_types: ["utmos"]
  predictor_args:
    utmos:
      fs: 16000
```

### configs/utmos_comprehensive.yaml
```yaml
- name: pseudo_mos
  predictor_types: ["utmos", "dnsmos", "plcmos"]
  predictor_args:
    utmos:
      fs: 16000
    dnsmos:
      fs: 16000
    plcmos:
      fs: 16000
```

### Creating Custom Configs

See [VERSA metrics documentation](versa/docs/supported_metrics.md) for available metrics.

Example custom config:
```yaml
# Custom evaluation config
- name: pseudo_mos
  predictor_types: ["utmos", "dnsmos"]
  predictor_args:
    utmos:
      fs: 16000
    dnsmos:
      fs: 16000

- name: speaker_similarity
  predictor_types: ["speaker_cosine"]
  predictor_args:
    speaker_cosine:
      backend: "wespeaker"
```

## Output Structure

After running evaluation, the output directory contains:

```
eval_results/
├── gt.scp                  # Ground truth file list
├── pred.scp               # Predicted file list
├── results.json           # Per-utterance scores
├── statistics.json        # Aggregated statistics
├── summary.txt           # Human-readable summary
└── evaluation.log        # Detailed logs
```

### results.json
Per-utterance scores:
```json
{
  "utt_id_001": {
    "utmos": 4.25
  },
  "utt_id_002": {
    "utmos": 3.87
  }
}
```

### statistics.json
Aggregate statistics:
```json
{
  "utmos": {
    "mean": 4.05,
    "min": 2.34,
    "max": 4.78,
    "count": 9113
  }
}
```

### summary.txt
Human-readable summary with all metrics and statistics.

## Metrics

### Currently Supported

#### UTMOS (Universal Test MOS)
- **Description**: Predicts Mean Opinion Score for audio naturalness
- **Range**: 1.0 (poor) to 5.0 (excellent)
- **Reference**: [UTMOS Paper](https://arxiv.org/abs/2204.02152)
- **Use Case**: Overall audio quality assessment

### Future Extensions

The pipeline is designed to easily add more metrics. Some planned additions:

- **PESQ**: Perceptual Evaluation of Speech Quality
- **STOI**: Short-Time Objective Intelligibility
- **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio
- **Speaker Similarity**: Cosine similarity of speaker embeddings
- **WER**: Word Error Rate (requires transcription)
- **Prosody Metrics**: F0 RMSE, duration RMSE

To add a new metric, create a config file in `configs/` following VERSA's format.

## Examples

### Example 1: Evaluate Generated Audio

```bash
# Generate audio (if not already done)
./generate_titw.sh

# Evaluate with UTMOS
./eval_versa.sh

# View results
cat eval_results/summary.txt
```

### Example 2: Evaluate Subset of Files

```bash
# Create subset SCP files
head -100 eval_results/gt.scp > subset_gt.scp
head -100 eval_results/pred.scp > subset_pred.scp

# Evaluate subset
python eval_versa.py \
    --gt_scp subset_gt.scp \
    --pred_scp subset_pred.scp \
    --output_dir eval_results_subset \
    --config configs/utmos.yaml
```

### Example 3: Compare Multiple Models

```bash
# Evaluate model 1
PRED_DIR=model1_output ./eval_versa.sh
mv eval_results eval_results_model1

# Evaluate model 2
PRED_DIR=model2_output ./eval_versa.sh
mv eval_results eval_results_model2

# Compare results
echo "Model 1 UTMOS:"
jq '.utmos.mean' eval_results_model1/statistics.json

echo "Model 2 UTMOS:"
jq '.utmos.mean' eval_results_model2/statistics.json
```

## Troubleshooting

### VERSA Script Not Found
```
Error: VERSA scorer script not found: versa/versa/bin/scorer.py
```
**Solution**: Check that the `versa/` directory is present and properly installed.

### No Common Files Found
```
ERROR: No common utterance IDs found between GT and predicted files!
```
**Solution**: Ensure file names match between GT and predicted directories (excluding path).

### Import Errors for Metrics
```
ImportError: No module named 'xxx'
```
**Solution**: Install metric-specific dependencies:
```bash
cd versa
python tools/install_<metric>.py
```

### Memory Issues
For large datasets, process in batches:
```bash
# Split file lists
split -l 1000 eval_results/gt.scp gt_batch_
split -l 1000 eval_results/pred.scp pred_batch_

# Process each batch
for i in gt_batch_*; do
    pred_batch="${i/gt_/pred_}"
    batch_num="${i#gt_batch_}"
    python eval_versa.py \
        --gt_scp "$i" \
        --pred_scp "$pred_batch" \
        --output_dir "eval_results_batch_${batch_num}" \
        --config configs/utmos.yaml
done
```

## Integration with Other Tools

### ESPnet Integration
The SCP file format is compatible with ESPnet:
```bash
# Use with ESPnet scorer
asr_scorer.py \
    --hyp eval_results/pred.scp \
    --ref eval_results/gt.scp
```

### Kaldi Integration
Compatible with Kaldi-style ARK files:
```bash
python eval_versa.py \
    --gt gt.scp \
    --pred pred.scp \
    --config configs/utmos.yaml \
    --versa_dir versa
# Add: --io kaldi (if using ARK format)
```

## Performance Notes

- **UTMOS**: ~0.5-1 second per utterance on GPU, ~2-5 seconds on CPU
- **Memory**: ~2-4GB for model loading
- **Batch Processing**: Recommended for datasets >10K utterances

## References

- **VERSA Toolkit**: [GitHub](https://github.com/wavlab-speech/versa)
- **UTMOS Paper**: [ArXiv](https://arxiv.org/abs/2204.02152)
- **CosyVoice**: [GitHub](https://github.com/FunAudioLLM/CosyVoice)

## Support

For issues with:
- **This pipeline**: Check logs in `eval_results/evaluation.log`
- **VERSA toolkit**: See [VERSA documentation](versa/README.md)
- **Metrics**: See [VERSA metrics guide](versa/docs/supported_metrics.md)
