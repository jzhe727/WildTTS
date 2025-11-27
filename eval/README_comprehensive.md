# Comprehensive VERSA Evaluation Configuration

This configuration file (`configs/comprehensive.yaml`) provides a complete evaluation setup with multiple metrics for assessing TTS-generated audio quality.

## Included Metrics

### 1. MCD (Mel Cepstral Distortion)
- **Purpose**: Measures spectral distance between generated and reference audio
- **Output Metrics**:
  - `mcd`: Mel cepstral distortion (lower is better)
  - `f0_corr`: F0 correlation (range: -1 to 1, higher is better)
  - `f0_rmse`: F0 root mean square error (lower is better)
- **Reference**: Standard objective metric for speech quality

### 2. UTMOS (UTokyo-SaruLab MOS)
- **Purpose**: Predicts Mean Opinion Score without human listeners
- **Output Metrics**:
  - `utmos`: MOS prediction (range: 1-5, higher is better)
- **Reference**: [Paper](https://arxiv.org/abs/2204.02152)
- **Model**: Uses UTokyo's pre-trained model

### 3. DNSMOS (Deep Noise Suppression MOS)
- **Purpose**: Predicts multiple quality dimensions
- **Output Metrics**:
  - `dnsmos_ovrl`: Overall quality (range: 1-5, higher is better)
  - `dnsmos_sig`: Signal quality
  - `dnsmos_bak`: Background quality
- **Reference**: [Microsoft's DNSMOS](https://github.com/microsoft/DNS-Challenge)

### 4. WER (Word Error Rate) with Whisper-Large
- **Purpose**: Measures transcription accuracy and intelligibility
- **Model**: OpenAI Whisper Large
- **Output Metrics**:
  - `whisper_hyp_text`: Transcribed hypothesis text
  - `ref_text`: Reference text (after cleaning)
  - `whisper_wer_delete`: Deletion errors
  - `whisper_wer_insert`: Insertion errors
  - `whisper_wer_replace`: Replacement errors
  - `whisper_wer_equal`: Correct word count
  - `whisper_cer_*`: Character-level equivalents
- **Note**: Corpus-level WER/CER requires manual aggregation across all utterances

### 5. Speaker Similarity
- **Purpose**: Measures speaker identity preservation
- **Model**: ESPnet voxcelebs12_rawnet3 (default)
- **Output Metrics**:
  - `spk_similarity`: Cosine similarity (range: -1 to 1, higher is better)
- **Alternative Models**: Can use WavLM-based models by specifying different `model_tag`
  - Example: `espnet/voxcelebs12_wavlm` (if available)
  - Check [ESPnet HuggingFace](https://huggingface.co/espnet) for available models

## Usage

### Run Comprehensive Evaluation
```bash
# Using the shell script (recommended)
./eval_versa.sh comprehensive

# Or directly with Python
python eval_versa.py \
    --config configs/comprehensive.yaml \
    --gt_dir titw-test \
    --pred_dir titw_generated \
    --output_dir eval_results_comprehensive
```

### Custom Ground Truth and Prediction Directories
```bash
GT_DIR=custom_gt PRED_DIR=custom_pred ./eval_versa.sh comprehensive
```

### Understanding the Results

Results are saved in JSON format with per-utterance scores:
```json
{
  "utt_001": {
    "mcd": 5.23,
    "f0_corr": 0.87,
    "f0_rmse": 12.34,
    "utmos": 4.21,
    "dnsmos_ovrl": 4.15,
    "dnsmos_sig": 4.20,
    "dnsmos_bak": 4.18,
    "whisper_hyp_text": "transcribed text",
    "whisper_wer_delete": 0,
    "whisper_wer_insert": 1,
    "whisper_wer_replace": 0,
    "whisper_wer_equal": 10,
    "spk_similarity": 0.92
  }
}
```

Statistics summary includes:
- Mean, min, max for each metric
- Count of evaluated utterances

## Metric Interpretation Guide

### Good Quality Indicators
- **MCD**: < 6.0 (lower is better)
- **F0 Correlation**: > 0.80 (higher is better)
- **UTMOS/DNSMOS**: > 4.0 (range: 1-5)
- **Speaker Similarity**: > 0.80 (range: -1 to 1)
- **WER**: < 10% (calculate as: (delete + insert + replace) / (equal + replace + delete) * 100)

### Excellent Quality Indicators
- **MCD**: < 4.0
- **F0 Correlation**: > 0.90
- **UTMOS/DNSMOS**: > 4.5
- **Speaker Similarity**: > 0.90
- **WER**: < 5%

## Requirements

The comprehensive evaluation requires additional dependencies:

```bash
# Install VERSA with required dependencies
cd versa
pip install -e .

# Install Whisper (for WER)
pip install openai-whisper

# Install ESPnet (for speaker similarity)
pip install espnet espnet_model_zoo

# Install other dependencies as needed
# Follow VERSA documentation: versa/docs/supported_metrics.md
```

## Customization

### Using Different Models

#### WavLM-based Speaker Model
Edit `configs/comprehensive.yaml`:
```yaml
- name: speaker
  model_tag: espnet/voxcelebs12_wavlm  # or other WavLM-based model
```

#### Different Whisper Model
```yaml
- name: whisper_wer
  model_tag: large-v3  # or: base, small, medium, large, large-v2
  beam_size: 5
  text_cleaner: whisper_basic
```

### Adding More DNSMOS Variants
```yaml
- name: pseudo_mos
  predictor_types: ["utmos", "dnsmos", "plcmos", "dnsmos_pro_bvcc"]
  predictor_args:
    utmos:
      fs: 16000
    dnsmos:
      fs: 16000
    plcmos:
      fs: 16000
```

## Troubleshooting

### Model Download Issues
- Models are automatically downloaded on first use
- They are cached in `~/.cache/huggingface/` and `~/.cache/espnet/`
- Ensure internet connectivity for first run

### Memory Issues
- Evaluation processes all files sequentially
- For large datasets, consider splitting into batches
- GPU is recommended for faster evaluation (especially for Whisper)

### WER Requires Text Reference
- WER calculation needs reference transcriptions
- Provide via `--text` argument pointing to a Kaldi-style text file
- Format: `utterance_id transcription text here`

## References

1. **MCD**: [Speech Signal Processing Toolkit (SPTK)](http://sp-tk.sourceforge.net/)
2. **UTMOS**: [UTokyo-SaruLab System for VoiceMOS Challenge 2022](https://arxiv.org/abs/2204.02152)
3. **DNSMOS**: [DNSMOS: A Non-Intrusive Perceptual Objective Speech Quality metric](https://arxiv.org/abs/2010.15258)
4. **Whisper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
5. **Speaker Similarity**: [ESPnet-SPK: Full Pipeline Speaker Embedding Toolkit](https://arxiv.org/abs/2401.17230)
6. **VERSA**: [VERSA: A Comprehensive Toolkit for Evaluating Speech and Audio](https://arxiv.org/abs/2412.17667)

## Contact & Support

For issues specific to:
- **VERSA toolkit**: Check [VERSA GitHub Issues](https://github.com/wavlab-speech/versa/issues)
- **This evaluation pipeline**: Check the eval/ directory documentation
- **Metric interpretations**: Refer to the original papers listed above
