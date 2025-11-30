#!/bin/bash
# VERSA Evaluation Pipeline Shell Script
# 
# This script runs the VERSA evaluation pipeline for TTS generated audio.
# It can be easily extended to support additional metrics in the future.
#
# Usage:
#   ./eval_versa.sh                    # Use default settings (UTMOS only)
#   ./eval_versa.sh comprehensive      # Use comprehensive config with multiple metrics
#   ./eval_versa.sh custom /path/to/config.yaml /path/to/output

set -e  # Exit on error

# Default settings
GT_DIR="${GT_DIR:-titw-test}"
PRED_DIR="${PRED_DIR:-titw_generated}"
OUTPUT_SUFFIX="results"
CONFIG="configs/utmos.yaml"
VERSA_DIR="versa"
TEXT_FILE=""  # Optional text file for WER
VERBOSE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        comprehensive)
            CONFIG="configs/comprehensive.yaml"
            OUTPUT_SUFFIX="results_comp"
            # Auto-detect text file for WER
            if [ -f "titw-test/metadata/text" ]; then
                TEXT_FILE="titw-test/metadata/text"
            fi
            shift
            ;;
        custom)
            shift
            if [ -n "$1" ] && [ "$1" != "-v" ] && [ "$1" != "--verbose" ]; then
                CONFIG="$1"
                shift
            fi
            if [ -n "$1" ] && [ "$1" != "-v" ] && [ "$1" != "--verbose" ]; then
                OUTPUT_DIR="$1"
                shift
            fi
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            echo "VERSA Evaluation Pipeline"
            echo ""
            echo "Usage:"
            echo "  ./eval_versa.sh [MODE] [OPTIONS]"
            echo ""
            echo "Modes:"
            echo "  (default)       - Run UTMOS evaluation only"
            echo "  comprehensive   - Run comprehensive evaluation with multiple metrics"
            echo "  custom CONFIG OUTPUT - Use custom config and output directory"
            echo ""
            echo "Options:"
            echo "  -v, --verbose   - Enable verbose logging"
            echo "  -h, --help      - Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  GT_DIR          - Ground truth directory (default: titw-test)"
            echo "  PRED_DIR        - Predicted audio directory (default: titw_generated)"
            echo ""
            echo "Examples:"
            echo "  ./eval_versa.sh"
            echo "  ./eval_versa.sh comprehensive"
            echo "  ./eval_versa.sh custom configs/my_config.yaml results/my_eval"
            echo "  ./eval_versa.sh comprehensive -v"
            echo "  GT_DIR=custom_gt PRED_DIR=custom_pred ./eval_versa.sh"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if required files exist
if [ ! -d "$GT_DIR" ]; then
    echo "Error: Ground truth directory not found: $GT_DIR"
    exit 1
fi

if [ ! -d "$PRED_DIR" ]; then
    echo "Error: Predicted directory not found: $PRED_DIR"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

if [ ! -d "$VERSA_DIR" ]; then
    echo "Error: VERSA directory not found: $VERSA_DIR"
    exit 1
fi

# if output dir not set, create based on prediction dir and suffix
if [ -z "$OUTPUT_DIR" ]; then
    PRED_BASENAME=$(basename "$PRED_DIR")
    OUTPUT_DIR="${PRED_BASENAME}_${OUTPUT_SUFFIX}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Display configuration
echo "========================================"
echo "VERSA Evaluation Pipeline"
echo "========================================"
echo "Ground Truth:  $GT_DIR"
echo "Predicted:     $PRED_DIR"
echo "Config:        $CONFIG"
echo "Output:        $OUTPUT_DIR"
echo "VERSA Dir:     $VERSA_DIR"
echo "========================================"
echo ""

# Run evaluation
echo "Starting evaluation..."
EVAL_CMD="python eval_versa.py \
    --gt_dir \"$GT_DIR\" \
    --pred_dir \"$PRED_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --config \"$CONFIG\" \
    --versa_dir \"$VERSA_DIR\""

# Add text file if provided
if [ -n "$TEXT_FILE" ]; then
    echo "Using text file: $TEXT_FILE"
    EVAL_CMD="$EVAL_CMD --text \"$TEXT_FILE\""
fi

# Add verbose flag
if [ -n "$VERBOSE" ]; then
    EVAL_CMD="$EVAL_CMD $VERBOSE"
fi

eval $EVAL_CMD

# Check if evaluation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Evaluation completed successfully!"
    echo "========================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    
    # Display summary if it exists
    if [ -f "$OUTPUT_DIR/summary.txt" ]; then
        echo "Summary:"
        echo "--------"
        cat "$OUTPUT_DIR/summary.txt"
        echo ""
    fi
    
    # Display statistics if they exist
    if [ -f "$OUTPUT_DIR/statistics.json" ]; then
        echo "Detailed statistics: $OUTPUT_DIR/statistics.json"
    fi
    
    # Display results file location
    if [ -f "$OUTPUT_DIR/results.json" ]; then
        echo "Full results: $OUTPUT_DIR/results.json"
    fi
else
    echo ""
    echo "========================================"
    echo "Evaluation failed!"
    echo "========================================"
    echo "Check the log file: $OUTPUT_DIR/evaluation.log"
    exit 1
fi
