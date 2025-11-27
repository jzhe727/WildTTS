# TITW TTS Generation Setup - Summary

## What Was Created

### 1. Main Generation Script: `generate_titw.py`

A comprehensive Python script that:
- Loads TITW test metadata (9113 utterances, 40 speakers)
- Uses CosyVoice2-0.5B for zero-shot voice cloning
- Generates one audio file per utterance
- Uses each speaker's first utterance as the voice prompt
- Maintains speaker identity across all generated utterances
- Includes robust error handling and progress tracking

**Key Features:**
- Zero-shot voice cloning (no fine-tuning needed)
- Speaker-consistent generation
- Versa-compatible output format
- Optional audio post-processing
- Detailed progress reporting

### 2. Shell Script: `generate_titw.sh`

A convenient wrapper script with:
- Easy command-line interface
- GPU acceleration options (--fp16, --jit)
- Post-processing control
- Input validation
- Usage examples

### 3. Documentation: `README_generate_titw.md`

Complete documentation including:
- Overview and features
- Data format specifications
- Usage examples (basic and advanced)
- Command-line options reference
- Output structure
- Troubleshooting guide
- Performance notes
- Integration with Versa toolkit

### 4. Verification Script: `verify_metadata.py`

Metadata validation tool that:
- Checks all required files exist
- Verifies utterance counts
- Validates speaker consistency
- Shows statistics and sample entries

## Metadata Verified

✅ **9113 utterances** from 40 speakers
✅ All metadata files consistent
✅ Average 227.8 utterances per speaker
✅ Min: 10 utterances, Max: 868 utterances per speaker

## How It Works

### Zero-Shot Voice Cloning Process

```
For each utterance:
1. Get speaker ID from utt2spk_test
2. Load speaker's first utterance as prompt:
   - prompt_wav: Original audio of first utterance
   - prompt_text: Transcript of first utterance
3. Generate target utterance:
   - tts_text: Same text as original (from text_test)
   - Uses prompt to maintain speaker voice
4. Save as: {utterance_id}.wav
```

### Example

```
Utterance: id10270-5r0dWxy17C8-00001-002
Speaker: id10270
Prompt (from first utterance):
  - Audio: test/id10270-5r0dWxy17C8-00001-001.wav
  - Text: "Very often I am, and then sometimes I'm not."
Target:
  - Text: "And when I catch myself"
Output:
  - File: titw_generated/id10270-5r0dWxy17C8-00001-002.wav
  - Voice: Matches id10270's original voice characteristics
```

## Usage

### Quick Start

```bash
cd /home/john.zheng1/ensf619/WildTTS/eval
./generate_titw.sh
```

### With GPU Acceleration

```bash
./generate_titw.sh --fp16 --jit --postprocess
```

### Direct Python

```bash
python generate_titw.py \
    --metadata_dir titw-test/metadata \
    --test_wav_dir titw-test \
    --output_dir titw_generated \
    --model_dir ../backend/CosyVoice/pretrained_models/CosyVoice2-0.5B \
    --postprocess_output
```

## Output Structure

```
WildTTS/eval/
├── generate_titw.py              # Main generation script
├── generate_titw.sh              # Shell wrapper script
├── verify_metadata.py            # Metadata verification tool
├── README_generate_titw.md       # Full documentation
├── titw-test/                    # Input data
│   ├── metadata/
│   │   ├── text_test            # Transcripts (9113 lines)
│   │   ├── wav_test.scp         # Audio paths (9113 lines)
│   │   ├── utt2spk_test         # Utterance→Speaker mapping
│   │   └── spk2utt_test         # Speaker→Utterances mapping
│   └── test/                     # Original audio files
│       └── *.wav                 # 9113 WAV files
└── titw_generated/               # Generated output (created)
    ├── id10270-*.wav             # Generated audio files
    ├── id10271-*.wav
    ├── ...
    └── failed_utterances.txt     # Any failures (if applicable)
```

## Versa Toolkit Compatibility

The generated audio files are **directly compatible** with Versa evaluation:

1. **File naming**: Matches original utterance IDs exactly
2. **Format**: Standard WAV files (PCM, 22050 Hz)
3. **Directory structure**: Flat structure for easy comparison
4. **Quality metrics**: Can evaluate with UTMOS, speaker similarity, etc.

### Evaluation Setup (Future)

```bash
cd versa
# Configure Versa to compare:
# - Reference: titw-test/test/*.wav
# - Generated: titw_generated/*.wav
# Run evaluation metrics
```

## Performance Estimates

- **Generation time**: 
  - GPU + JIT: ~2-3 seconds per utterance
  - GPU: ~4-5 seconds per utterance
  - CPU: ~10-20 seconds per utterance

- **Total time for 9113 files**:
  - GPU + JIT: ~5-8 hours
  - GPU: ~10-12 hours
  - CPU: ~25-50 hours

- **Disk space**: ~2-3 GB for all generated files

## Next Steps

### Immediate
1. ✅ Scripts created and verified
2. ✅ Metadata validated (9113 utterances, 40 speakers)
3. ⏳ Ready to run generation

### After Generation
1. Run Versa evaluation metrics
2. Analyze speaker-wise quality
3. Compare with original recordings
4. Identify any failure cases
5. Fine-tune parameters if needed

### Future Enhancements
- Batch processing for faster generation
- Multi-GPU support
- Prompt selection optimization
- Quality filtering
- Automated evaluation pipeline

## Files Created

1. ✅ `generate_titw.py` - Main generation script (479 lines)
2. ✅ `generate_titw.sh` - Shell wrapper (executable)
3. ✅ `README_generate_titw.md` - Full documentation
4. ✅ `verify_metadata.py` - Metadata verification tool
5. ✅ This summary document

## Dependencies

Required Python packages:
```
torch
torchaudio
librosa
tqdm
```

CosyVoice2 model:
```
backend/CosyVoice/pretrained_models/CosyVoice2-0.5B/
```

## Testing

Metadata verification passed:
```
✓ 9113 utterances from 40 speakers
✓ All metadata files consistent
✓ File format validated
✓ Speaker mappings verified
```

## Important Notes

1. **Zero-shot voice cloning**: No fine-tuning or training required
2. **Speaker consistency**: Uses first utterance as prompt for all speaker's utterances
3. **Text matching**: Generates the SAME text as the original (for evaluation purposes)
4. **No evaluation pipeline yet**: Generation only, evaluation to be set up separately
5. **Versa compatible**: Output format ready for Versa toolkit evaluation

## Questions or Issues?

See `README_generate_titw.md` for:
- Detailed usage instructions
- Troubleshooting guide
- Performance optimization tips
- Integration with Versa toolkit
