# Comprehensive VERSA Evaluation - Setup Complete ✅

## What Was Created

### 1. Comprehensive Configuration: `configs/comprehensive.yaml`

A complete evaluation configuration with 5 metric categories:

| Metric Category | Protocol | Measures | Output Variables |
|----------------|----------|----------|------------------|
| **MCD** | `mcd_f0` | Spectral and pitch accuracy | `mcd`, `f0_corr`, `f0_rmse` |
| **UTMOS** | `pseudo_mos` | Overall perceived quality | `utmos` |
| **DNSMOS** | `pseudo_mos` | Multi-dimensional quality | `dnsmos_ovrl`, `dnsmos_sig`, `dnsmos_bak` |
| **WER** | `whisper_wer` | Transcription accuracy | `whisper_wer_*`, `whisper_cer_*` |
| **Speaker Sim** | `speaker` | Speaker identity preservation | `spk_similarity` |

### 2. Updated Evaluation Scripts

**`eval_versa.py`** - Enhanced with:
- Text file support for WER calculation
- Automatic text file handling
- Compatible with all VERSA protocols

**`eval_versa.sh`** - Enhanced with:
- `comprehensive` mode for multi-metric evaluation
- Auto-detection of text file (`titw-test/metadata/text`)
- Improved argument handling

### 3. Documentation Files

- **`README_comprehensive.md`** - Detailed guide with metric explanations, customization options, and references
- **`QUICK_REFERENCE.md`** - Quick start guide with score interpretation and troubleshooting

## 🚀 Quick Start

```bash
# Run comprehensive evaluation
./eval_versa.sh comprehensive
```

That's it! The script will:
1. ✅ Find matching audio files between ground truth and predictions
2. ✅ Auto-detect text transcriptions for WER
3. ✅ Run all 5 metric categories
4. ✅ Generate comprehensive results

## 📁 Output Structure

```
eval_results_comprehensive/
├── results.json          # Per-utterance scores for all metrics
├── statistics.json       # Mean, min, max, count per metric
├── summary.txt          # Human-readable summary report
├── evaluation.log       # Detailed execution log
├── gt.scp              # Ground truth file list
└── pred.scp            # Predicted file list
```

## 🎯 Score Interpretation

### Good vs Excellent Thresholds

| Metric | Good | Excellent | Better |
|--------|------|-----------|--------|
| MCD | < 6.0 | < 4.0 | Lower |
| F0 Corr | > 0.80 | > 0.90 | Higher |
| UTMOS | > 4.0 | > 4.5 | Higher |
| DNSMOS | > 4.0 | > 4.5 | Higher |
| Spk Sim | > 0.80 | > 0.90 | Higher |
| WER | < 10% | < 5% | Lower |

## 🔧 Key Configuration Details

### Protocol Names (Following VERSA Documentation)

The config uses VERSA's official protocol names:
- ✅ `mcd_f0` - Includes both MCD and F0 metrics
- ✅ `pseudo_mos` - For UTMOS and DNSMOS predictors
- ✅ `whisper_wer` - OpenAI Whisper-based WER (using large model)
- ✅ `speaker` - ESPnet speaker embedding similarity

### Model Selections

1. **WER**: OpenAI Whisper Large
   - `model_tag: large`
   - Can be changed to: `base`, `small`, `medium`, `large-v3`

2. **Speaker Similarity**: ESPnet RawNet3 (default)
   - `model_tag: default` → `espnet/voxcelebs12_rawnet3`
   - WavLM alternative available: Check ESPnet HuggingFace for models

3. **UTMOS**: Uses pre-trained UTokyo model
4. **DNSMOS**: Uses Microsoft's pre-trained model
5. **MCD**: Uses SPTK (Speech Signal Processing Toolkit)

## 🔄 Extensibility

The pipeline is designed to be easily extended:

### Add More Metrics

Edit `configs/comprehensive.yaml` and add:

```yaml
# Example: Add PESQ
- name: pesq

# Example: Add STOI  
- name: stoi

# Example: Add more MOS predictors
- name: pseudo_mos
  predictor_types: ["utmos", "dnsmos", "plcmos", "dnsmos_pro_bvcc"]
```

See `versa/egs/speech.yaml` for all available metrics.

### Custom Evaluation Configs

```bash
# Create custom config
cp configs/comprehensive.yaml configs/my_custom.yaml
# Edit my_custom.yaml as needed

# Run with custom config
./eval_versa.sh custom configs/my_custom.yaml eval_results_custom
```

## ⚙️ System Requirements

### Minimum
- **RAM**: 8GB
- **Storage**: 5GB (for model cache)
- **Internet**: Required for first-time model downloads

### Recommended
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 11.8 or higher
- **Storage**: 10GB for model cache

### Model Downloads (First Run Only)
- Whisper Large: ~3GB
- UTMOS model: ~100MB
- DNSMOS model: ~30MB
- ESPnet speaker model: ~100MB

Models cached in:
- `~/.cache/huggingface/`
- `~/.cache/espnet/`

## 🐛 Common Issues & Solutions

### 1. Text File Not Found (WER fails)
```bash
# Verify text file exists
ls -lh titw-test/metadata/text

# If in different location, specify manually:
python eval_versa.py --config configs/comprehensive.yaml \
    --text /path/to/text/file \
    --output_dir eval_results_comprehensive
```

### 2. CUDA Out of Memory
```bash
# Option 1: Use CPU
export CUDA_VISIBLE_DEVICES=""
./eval_versa.sh comprehensive

# Option 2: Use smaller Whisper model
# Edit configs/comprehensive.yaml, change:
# model_tag: medium  (instead of large)
```

### 3. No Matching Files
- Ensure generated audio filenames match ground truth filenames (without extension)
- Check with: `ls titw-test/*.wav | wc -l` and `ls titw_generated/*.wav | wc -l`

### 4. Model Download Fails
```bash
# Manually download models first
python -c "import whisper; whisper.load_model('large')"
python -c "from espnet2.bin.spk_inference import Speech2Embedding; Speech2Embedding.from_pretrained('espnet/voxcelebs12_rawnet3')"
```

## 📊 Understanding WER Results

WER is provided as error counts per utterance. To calculate percentage:

```python
# Per utterance
deletions = result["whisper_wer_delete"]
insertions = result["whisper_wer_insert"]  
replacements = result["whisper_wer_replace"]
correct = result["whisper_wer_equal"]

total_ref = deletions + replacements + correct
wer = ((deletions + insertions + replacements) / total_ref) * 100

# Corpus-level (aggregate across all utterances first)
total_del = sum(all_deletions)
total_ins = sum(all_insertions)
total_rep = sum(all_replacements)
total_cor = sum(all_correct)
total_ref_words = total_del + total_rep + total_cor
corpus_wer = ((total_del + total_ins + total_rep) / total_ref_words) * 100
```

## 📚 Documentation References

1. **Quick Start**: `QUICK_REFERENCE.md`
2. **Detailed Guide**: `README_comprehensive.md`
3. **VERSA Official**: `versa/README.md`
4. **Available Metrics**: `versa/docs/supported_metrics.md`
5. **Example Configs**: `versa/egs/` directory

## 🎓 Metric Citations

- **MCD**: SPTK - http://sp-tk.sourceforge.net/
- **UTMOS**: https://arxiv.org/abs/2204.02152
- **DNSMOS**: https://arxiv.org/abs/2010.15258
- **Whisper**: https://arxiv.org/abs/2212.04356
- **Speaker Sim**: https://arxiv.org/abs/2401.17230
- **VERSA**: https://arxiv.org/abs/2412.17667

## ✅ Next Steps

1. **Run evaluation**: `./eval_versa.sh comprehensive`
2. **View summary**: `cat eval_results_comprehensive/summary.txt`
3. **Analyze results**: Check `eval_results_comprehensive/results.json`
4. **Iterate**: Improve your TTS model based on weak metrics
5. **Compare**: Run multiple experiments and compare results

## 💡 Tips

- **First run takes longer** due to model downloads
- **Use GPU** for 3-5x faster evaluation
- **Batch processing** for very large datasets
- **Keep logs** for debugging and reproducibility
- **Version control** your config files for different experiments
