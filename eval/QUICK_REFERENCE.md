# TITW TTS Generation - Quick Reference

## 📋 Task Summary
Generate 9113 synthetic speech files using CosyVoice2 zero-shot voice cloning for TITW test set (40 speakers).

## 🚀 Quick Start

```bash
cd /home/john.zheng1/ensf619/WildTTS/eval
./generate_titw.sh
```

## 📊 Dataset Info
- **Utterances**: 9113
- **Speakers**: 40
- **Avg per speaker**: 227.8 utterances
- **Format**: 16kHz WAV files

## 🛠️ Command Options

### Basic
```bash
./generate_titw.sh
```

### GPU Accelerated
```bash
./generate_titw.sh --fp16 --jit
```

### With Post-processing
```bash
./generate_titw.sh --postprocess
```

### Custom Output
```bash
./generate_titw.sh --output_dir my_output --speed 1.1
```

### Help
```bash
./generate_titw.sh --help
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `generate_titw.py` | Main generation script |
| `generate_titw.sh` | Shell wrapper (easy to use) |
| `verify_metadata.py` | Check metadata consistency |
| `README_generate_titw.md` | Full documentation |
| `SETUP_SUMMARY.md` | Detailed setup info |

## 📂 Directory Structure

```
eval/
├── generate_titw.py           # Main script
├── generate_titw.sh           # Shell wrapper
├── verify_metadata.py         # Verification
├── titw-test/                 # Input data
│   ├── metadata/              # Text, speaker info
│   └── test/                  # Original WAV files (9113)
└── titw_generated/            # Output (to be created)
    └── *.wav                  # Generated files (9113)
```

## ⚙️ How It Works

1. **Load speaker prompts**: First utterance from each speaker
2. **For each test utterance**:
   - Get speaker ID
   - Load speaker's prompt (audio + text)
   - Generate target text with that voice
   - Save output WAV file
3. **Report**: Success count, failures, timing

## 🎯 Zero-Shot Voice Cloning

```
Input:
  - Prompt audio: Speaker's first utterance (original)
  - Prompt text: Transcript of first utterance
  - Target text: Text to synthesize (same as original)

Output:
  - Generated audio: Target text in speaker's voice
  - Maintains speaker characteristics
```

## ⏱️ Estimated Time

| Configuration | Time per utterance | Total (9113 files) |
|---------------|--------------------|--------------------|
| GPU + JIT     | ~2-3 sec           | ~5-8 hours         |
| GPU           | ~4-5 sec           | ~10-12 hours       |
| CPU           | ~10-20 sec         | ~25-50 hours       |

## 💾 Disk Space
~2-3 GB for all generated files

## ✅ Metadata Verification

```bash
python verify_metadata.py
```

Expected output:
```
✓ 9113 utterances
✓ 40 speakers
✓ All files consistent
```

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Remove JIT flag
./generate_titw.sh --fp16
# Or use CPU (slower)
```

### Permission Denied
```bash
chmod +x generate_titw.sh
```

### Model Not Found
```bash
# Check model path
ls ../backend/CosyVoice/pretrained_models/CosyVoice2-0.5B/
```

## 📈 Output Format

Versa-compatible WAV files:
- **Naming**: `{utterance_id}.wav` (e.g., `id10270-5r0dWxy17C8-00001-001.wav`)
- **Format**: PCM WAV
- **Sample rate**: 22050 Hz
- **Channels**: 1 (mono)

## 🔗 Integration with Versa

Generated files are ready for Versa evaluation:
```bash
cd versa
# Configure Versa to compare:
# Reference: ../titw-test/test/*.wav
# Generated: ../titw_generated/*.wav
```

## 📚 More Info

- **Full docs**: `README_generate_titw.md`
- **Setup details**: `SETUP_SUMMARY.md`
- **CosyVoice**: `../backend/CosyVoice/`
- **Versa toolkit**: `versa/`

## ⚠️ Important Notes

1. ✅ No training/fine-tuning required (zero-shot)
2. ✅ Maintains speaker identity across utterances
3. ✅ Generates SAME text as original (for evaluation)
4. ⏳ Evaluation pipeline NOT included (generation only)
5. ✅ Output format compatible with Versa

## 🎓 Example Usage

```bash
# Step 1: Verify metadata
python verify_metadata.py

# Step 2: Generate (GPU with post-processing)
./generate_titw.sh --fp16 --jit --postprocess

# Step 3: Check output
ls titw_generated/ | wc -l  # Should show 9113

# Step 4: Check for failures (if any)
cat titw_generated/failed_utterances.txt
```

## 📞 Support

See `README_generate_titw.md` for:
- Detailed usage
- Command-line options
- Performance tips
- Troubleshooting guide
