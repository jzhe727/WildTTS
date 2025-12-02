#!/bin/bash
# Generate TITW samples using FishAudio
# Usage: ./generate_titw_fishaudio.sh [--enhance] [--no-transcript]

eval "$(conda shell.bash hook)"
conda activate wildtts

cd "$(dirname "$0")"

ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --enhance)
            ARGS="$ARGS --enhance_audio_quality"
            shift
            ;;
        --no-transcript)
            ARGS="$ARGS --no_transcript"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--enhance] [--no-transcript]"
            exit 1
            ;;
    esac
done

python generate_titw_fishaudio.py $ARGS
