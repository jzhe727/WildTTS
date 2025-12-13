#!/bin/bash
# Script to generate TTS audio for TITW test set using CosyVoice2

# Set working directory to script location
cd "$(dirname "$0")"

eval "$(conda shell.bash hook)"
conda activate wildtts

# Default parameters
METADATA_DIR="titw-test/metadata"
TEST_WAV_DIR="titw-test"
OUTPUT_DIR="titw_generated"
MODEL_DIR="../backend/CosyVoice/pretrained_models/CosyVoice2-0.5B"

# Parse command line arguments
USE_FP16=false
USE_JIT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;  

        --fp16)
            USE_FP16=true
            shift
            ;;
        --jit)
            USE_JIT=true
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fp16              Use FP16 precision (requires GPU)"
            echo "  --jit               Use JIT-optimized model (faster)"
            echo "  --output_dir DIR    Output directory (default: titw_generated)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Basic generation"
            echo "  $0 --fp16 --jit             # GPU-accelerated generation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python generate_titw.py"
CMD="$CMD --metadata_dir $METADATA_DIR"
CMD="$CMD --test_wav_dir $TEST_WAV_DIR"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --model_dir $MODEL_DIR"

if [ "$USE_FP16" = true ]; then
    CMD="$CMD --fp16"
fi

if [ "$USE_JIT" = true ]; then
    CMD="$CMD --load_jit"
fi

# Print configuration
echo "========================================"
echo "TITW TTS Generation"
echo "========================================"
echo "Metadata dir:  $METADATA_DIR"
echo "Test wav dir:  $TEST_WAV_DIR"
echo "Output dir:    $OUTPUT_DIR"
echo "Model dir:     $MODEL_DIR"
echo "FP16:          $USE_FP16"
echo "JIT:           $USE_JIT"
echo "========================================"
echo ""
echo "Command: $CMD"
echo ""

# Check if required files exist
if [ ! -d "$METADATA_DIR" ]; then
    echo "Error: Metadata directory not found: $METADATA_DIR"
    exit 1
fi

if [ ! -d "$TEST_WAV_DIR" ]; then
    echo "Error: Test wav directory not found: $TEST_WAV_DIR"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Run generation
echo "Starting generation..."
echo ""
$CMD

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Generation completed successfully!"
    echo "Output directory: $OUTPUT_DIR"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Generation failed!"
    echo "========================================"
    exit 1
fi
