#!/bin/bash
# Evaluate 5 VBench dimensions on custom videos and save results to the video folder.
#
# Usage:
#   bash run_eval.sh /path/to/video_folder [--ngpus N]
#
# The video folder should contain .mp4 or .gif files.
# Prompts are extracted from filenames (e.g. "a dog running-0.mp4" -> "a dog running").
# Results JSON will be saved into the video folder itself.

set -e

# ── Parse arguments ──────────────────────────────────────────────
if [ -z "$1" ]; then
    echo "Usage: bash run_eval.sh <video_folder> [--ngpus N]"
    exit 1
fi

VIDEO_DIR="$(realpath "$1")"
shift

NGPUS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ngpus) NGPUS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ ! -d "$VIDEO_DIR" ]; then
    echo "Error: $VIDEO_DIR is not a directory."
    exit 1
fi

DIMENSIONS="aesthetic_quality subject_consistency overall_consistency motion_smoothness dynamic_degree"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$VIDEO_DIR/vbench_results"

echo "==> Video folder : $VIDEO_DIR"
echo "==> Dimensions   : $DIMENSIONS"
echo "==> GPUs         : $NGPUS"
echo "==> Output       : $OUTPUT_DIR"
echo ""

# ── Run evaluation ───────────────────────────────────────────────
cd "$SCRIPT_DIR"

if [ "$NGPUS" -gt 1 ]; then
    torchrun --nproc_per_node="$NGPUS" -m vbench.launch.evaluate \
        --videos_path "$VIDEO_DIR" \
        --dimension $DIMENSIONS \
        --mode custom_input \
        --output_path "$OUTPUT_DIR" \
        --load_ckpt_from_local True
else
    python -m vbench.launch.evaluate \
        --videos_path "$VIDEO_DIR" \
        --dimension $DIMENSIONS \
        --mode custom_input \
        --output_path "$OUTPUT_DIR" \
        --load_ckpt_from_local True
fi

echo ""
echo "==> Evaluation complete! Results saved to:"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || echo "(no JSON found)"
