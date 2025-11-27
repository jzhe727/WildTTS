# TITW Test Set TTS Generation

This script generates synthetic speech for the TITW (Text-In-The-Wild) test set using CosyVoice2 zero-shot voice cloning.

## Overview

The script processes **9113 test utterances** from 40 unique speakers, generating audio files that maintain speaker identity through zero-shot voice cloning. Each generated file can be directly used with the Versa evaluation toolkit.

## Features

- **Zero-shot voice cloning**: Uses the first utterance from each speaker as a prompt
- **Speaker-consistent generation**: Maintains voice characteristics across all utterances
- **Versa-compatible output**: Generated files follow the naming convention required by Versa
- **Robust error handling**: Tracks and reports failed generations

## Data Format

### Input Files (in `titw-test/metadata/`)

1. **text_test**: Utterance transcripts
   ```
   id10270-5r0dWxy17C8-00001-001 Very often I am, and then sometimes I'm not.
   ```

2. **wav_test.scp**: Original audio file paths
   ```
   id10270-5r0dWxy17C8-00001-001 test/id10270-5r0dWxy17C8-00001-001.wav
   ```

3. **utt2spk_test**: Utterance to speaker mapping
   ```
   id10270-5r0dWxy17C8-00001-001 id10270
   ```

4. **spk2utt_test**: Speaker to utterance mapping (used to select prompt)
   ```
   id10270 id10270-5r0dWxy17C8-00001-001 id10270-5r0dWxy17C8-00001-002 ...
   ```

## Usage

### Basic Usage

```bash
cd /home/john.zheng1/ensf619/WildTTS/eval
python generate_titw.py
```

This will:
- Load metadata from `titw-test/metadata/`
- Use CosyVoice2-0.5B model from `../backend/CosyVoice/pretrained_models/CosyVoice2-0.5B`
- Generate 9113 audio files to `titw_generated/`

### Advanced Usage

```bash
python generate_titw.py \
    --metadata_dir titw-test/metadata \
    --test_wav_dir titw-test \
    --output_dir titw_generated \
    --model_dir ../backend/CosyVoice/pretrained_models/CosyVoice2-0.5B \
    --speed 1.0 \
    --postprocess_output
```

### Command Line Options

#### Path Options
- `--metadata_dir`: Path to metadata directory (default: `titw-test/metadata`)
- `--test_wav_dir`: Directory containing test WAV files (default: `titw-test`)
- `--output_dir`: Output directory for generated files (default: `titw_generated`)
- `--model_dir`: Path to CosyVoice2 model (default: `../backend/CosyVoice/pretrained_models/CosyVoice2-0.5B`)

#### Model Options
- `--load_jit`: Load JIT-optimized model (faster inference)
- `--load_trt`: Load TensorRT-optimized model (requires TensorRT)
- `--fp16`: Use FP16 precision (faster, requires GPU)

#### Generation Options
- `--speed`: Speed factor for generation (default: 1.0)
- `--postprocess_prompt`: Apply audio post-processing to prompt audio
- `--postprocess_output`: Apply audio post-processing to generated audio (trim silence, normalize)

### Example with GPU Acceleration

```bash
python generate_titw.py \
    --fp16 \
    --load_jit \
    --postprocess_output \
    --output_dir titw_generated_fp16
```

## Output

### Generated Files

The script creates:
1. **Audio files**: `{utterance_id}.wav` in the output directory
   - Example: `id10270-5r0dWxy17C8-00001-001.wav`
   - Format: 16-bit PCM WAV
   - Sample rate: 22050 Hz (CosyVoice2 default)

2. **Failed utterances list** (if any): `failed_utterances.txt`
   - Contains utterance IDs that failed generation
   - Useful for debugging or re-running specific utterances

### Output Structure

```
titw_generated/
├── id10270-5r0dWxy17C8-00001-001.wav
├── id10270-5r0dWxy17C8-00001-002.wav
├── ...
├── id10281-qBTOwaSOS-M-00003-001.wav
└── failed_utterances.txt (if any failures)
```

## Process Flow

1. **Load Metadata**: Parse text_test, wav_test.scp, utt2spk_test, spk2utt_test
2. **Extract Speaker Prompts**: For each speaker, use their first utterance as the voice prompt
3. **Initialize Model**: Load CosyVoice2 with specified options
4. **Generate Audio**: For each utterance:
   - Load speaker's prompt audio
   - Run zero-shot inference with target text
   - Save generated audio
5. **Report Results**: Print summary and save failed utterances list

## Zero-Shot Voice Cloning

The script uses **zero-shot voice cloning** via CosyVoice2:

1. **Prompt Audio**: First utterance from each speaker (original recording)
2. **Prompt Text**: Transcript of the prompt audio
3. **Target Text**: Text to synthesize (same as original utterance text)
4. **Output**: Generated audio that matches the speaker's voice characteristics

This approach maintains speaker identity while generating the entire test set.

## Compatibility with Versa Toolkit

The generated audio files are compatible with the Versa speech evaluation toolkit:

- **File naming**: Matches original utterance IDs
- **Format**: Standard WAV files
- **Usage**: Can be evaluated against reference audio using Versa metrics

To evaluate with Versa:

```bash
cd versa
# Configure to compare generated audio against reference
# See Versa documentation for specific evaluation commands
```

## Dependencies

Ensure the following are installed:

```bash
pip install torch torchaudio librosa tqdm
```

CosyVoice dependencies are included in the backend setup.

## Troubleshooting

### CUDA Out of Memory

If you encounter GPU memory errors:
- Remove `--load_jit` flag
- Reduce batch processing (script processes one at a time by default)
- Use CPU inference (slower but no memory limits)

### Missing Speaker Prompts

If speakers have no prompt audio:
- Check that `spk2utt_test` contains all speakers
- Verify that prompt WAV files exist in `test_wav_dir`

### Failed Generations

Check `failed_utterances.txt` for specific utterance IDs that failed.
Common causes:
- Corrupted input audio
- Very long texts (exceeds model limits)
- Audio format issues

## Performance Notes

- **Generation time**: ~2-5 seconds per utterance (GPU), ~10-20 seconds (CPU)
- **Total time**: Approximately 5-10 hours for all 9113 files (GPU with JIT)
- **Disk space**: ~2-3 GB for all generated audio files

## Next Steps

After generation, you can:
1. Evaluate using Versa toolkit metrics (UTMOS, speaker similarity, etc.)
2. Compare generated audio quality across different speakers
3. Analyze failure cases if any
4. Fine-tune generation parameters based on evaluation results

## References

- CosyVoice2: https://github.com/FunAudioLLM/CosyVoice
- Versa Toolkit: See `versa/` directory
- TITW Dataset: See `titw-test/README.md`
