# Pre-Generation Checklist

## Before Running Generation

### ✅ 1. Environment Check

```bash
# Check Python version (3.8+)
python --version

# Check if in correct directory
pwd
# Should show: /home/john.zheng1/ensf619/WildTTS/eval

# Check CUDA availability (optional, for GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### ✅ 2. Verify Metadata

```bash
# Run metadata verification
python verify_metadata.py
```

Expected output:
- ✓ All files have 9113 utterances
- ✓ Consistent speaker IDs: 40 speakers
- ✓ Metadata verification successful!

### ✅ 3. Check Model Files

```bash
# Check if model directory exists
ls -lh ../backend/CosyVoice/pretrained_models/CosyVoice2-0.5B/

# Should see:
# - cosyvoice2.yaml
# - llm.pt
# - flow.pt
# - hift.pt
# - speech_tokenizer_v2.onnx
# - campplus.onnx
```

### ✅ 4. Verify Test Audio Files

```bash
# Check if test audio directory exists
ls titw-test/test/ | head -5

# Count files
ls titw-test/test/*.wav | wc -l
# Should show: 9113
```

### ✅ 5. Check Dependencies

```bash
# Check required Python packages
python -c "import torch; print('torch:', torch.__version__)"
python -c "import torchaudio; print('torchaudio:', torchaudio.__version__)"
python -c "import librosa; print('librosa:', librosa.__version__)"
python -c "import tqdm; print('tqdm installed')"
```

If any missing:
```bash
pip install torch torchaudio librosa tqdm
```

### ✅ 6. Disk Space Check

```bash
# Check available disk space (need ~3GB)
df -h .
```

### ✅ 7. Test Run (Optional)

Create a small test with first 10 utterances:

```bash
# Create test script
cat > test_generation.py << 'EOF'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '../backend/CosyVoice')
sys.path.insert(0, '../backend/CosyVoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2

print("Loading model...")
model_dir = "../backend/CosyVoice/pretrained_models/CosyVoice2-0.5B"
cosyvoice = CosyVoice2(model_dir)
print("Model loaded successfully!")
print(f"Sample rate: {cosyvoice.sample_rate}")
EOF

python test_generation.py
```

---

## Pre-Run Configuration

### Choose Your Configuration

**Option A: Fast GPU Generation (Recommended)**
```bash
./generate_titw.sh --fp16 --jit --postprocess
```
- Time: ~5-8 hours
- Requires: GPU with 8GB+ VRAM
- Quality: High

**Option B: Standard GPU Generation**
```bash
./generate_titw.sh --postprocess
```
- Time: ~10-12 hours
- Requires: GPU with 6GB+ VRAM
- Quality: High

**Option C: CPU Generation**
```bash
./generate_titw.sh --postprocess
```
- Time: ~25-50 hours
- Requires: CPU only
- Quality: High

### Prepare for Long Run

```bash
# Use screen or tmux to prevent interruption
screen -S titw_generation
# or
tmux new -s titw_generation

# Then run your chosen command
./generate_titw.sh --fp16 --jit --postprocess

# Detach: Ctrl+A, D (screen) or Ctrl+B, D (tmux)
```

---

## During Generation

### Monitor Progress

The script will show:
```
Processing utterances: XX% |████████░░| XXXX/9113
```

### Check Partial Output

```bash
# In another terminal
ls titw_generated/*.wav | wc -l
```

### Check Logs (if using screen/tmux)

```bash
# Reattach to session
screen -r titw_generation
# or
tmux attach -t titw_generation
```

---

## After Generation

### ✅ 1. Verify Output Count

```bash
# Count generated files
ls titw_generated/*.wav | wc -l
# Should show: 9113 (or close to it)
```

### ✅ 2. Check for Failures

```bash
# Check if failed utterances file exists
if [ -f titw_generated/failed_utterances.txt ]; then
    echo "Some utterances failed:"
    wc -l titw_generated/failed_utterances.txt
    head titw_generated/failed_utterances.txt
else
    echo "All utterances generated successfully!"
fi
```

### ✅ 3. Verify File Sizes

```bash
# Check average file size
ls -lh titw_generated/*.wav | awk '{sum+=$5; count++} END {print "Average:", sum/count/1024, "KB"}'

# Find any suspiciously small files
find titw_generated -name "*.wav" -size -10k
```

### ✅ 4. Test Random Samples

```bash
# Play a random sample (if audio playback available)
FILE=$(ls titw_generated/*.wav | shuf -n 1)
echo "Playing: $FILE"
# Use your audio player, e.g.:
# play $FILE
# aplay $FILE
# ffplay $FILE
```

### ✅ 5. Prepare for Versa Evaluation

```bash
# Create reference list
ls titw-test/test/*.wav | sort > reference_files.txt

# Create generated list
ls titw_generated/*.wav | sort > generated_files.txt

# Check counts match
wc -l reference_files.txt generated_files.txt
```

---

## Troubleshooting Checklist

### Issue: CUDA Out of Memory

- [ ] Try without JIT: `./generate_titw.sh --fp16`
- [ ] Use CPU: Remove `--fp16` flag
- [ ] Close other GPU programs
- [ ] Check GPU memory: `nvidia-smi`

### Issue: Model Loading Failed

- [ ] Check model directory exists
- [ ] Verify all model files present
- [ ] Check file permissions
- [ ] Verify CosyVoice installation

### Issue: Slow Generation

- [ ] Confirm GPU is being used: Check nvidia-smi during run
- [ ] Try with JIT optimization: `--jit`
- [ ] Ensure no other heavy processes running

### Issue: Many Failed Utterances

- [ ] Check if specific speakers failing
- [ ] Verify input audio files readable
- [ ] Check for corrupted audio files
- [ ] Review error messages in output

---

## Quick Checklist Summary

Before running:
- [ ] Environment verified (Python, CUDA)
- [ ] Metadata verified (9113 utterances, 40 speakers)
- [ ] Model files present
- [ ] Test audio files accessible (9113 files)
- [ ] Dependencies installed
- [ ] Sufficient disk space (~3GB)
- [ ] Screen/tmux session started (for long runs)

After running:
- [ ] Output count verified (~9113 files)
- [ ] Failures checked (if any)
- [ ] File sizes reasonable
- [ ] Random samples tested
- [ ] Ready for Versa evaluation

---

## Ready to Generate?

```bash
# Final check
python verify_metadata.py

# Start generation
./generate_titw.sh --fp16 --jit --postprocess

# Or with screen
screen -S titw_gen
./generate_titw.sh --fp16 --jit --postprocess
# Ctrl+A, D to detach
```

Good luck! 🚀
