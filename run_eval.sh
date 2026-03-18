#!/bin/bash
# Evaluate 5 VBench dimensions on custom videos and save results to the video folder.
#
# Usage:
#   bash run_eval.sh <video_folder> <prompt_txt> [--ngpus N]
#
# Arguments:
#   video_folder  - folder containing .mp4/.gif video files
#   prompt_txt    - a text file with one prompt per line, matched to videos
#                   by sorted order (line 1 -> 1st video alphabetically, etc.)
#
# Results JSON will be saved into <video_folder>/vbench_results/

set -e

# ── Parse arguments ──────────────────────────────────────────────
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash run_eval.sh <video_folder> <prompt_txt> [--ngpus N]"
    echo ""
    echo "  video_folder  folder with .mp4/.gif files"
    echo "  prompt_txt    text file, one prompt per line (matched by sorted filename order)"
    exit 1
fi

VIDEO_DIR="$(realpath "$1")"
PROMPT_TXT="$(realpath "$2")"
shift 2

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
if [ ! -f "$PROMPT_TXT" ]; then
    echo "Error: $PROMPT_TXT is not a file."
    exit 1
fi

DIMENSIONS="aesthetic_quality subject_consistency overall_consistency motion_smoothness dynamic_degree"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$VIDEO_DIR/vbench_results"

# ── Build prompt JSON ────────────────────────────────────────────
# VBench --prompt_file expects: {"<video_folder>/<filename>": "prompt", ...}
# We match prompts to videos by sorted filename order.
PROMPT_JSON="$OUTPUT_DIR/prompt_map.json"
mkdir -p "$OUTPUT_DIR"

python3 -c "
import os, json, sys

video_dir = sys.argv[1]
prompt_txt = sys.argv[2]
output_json = sys.argv[3]

# collect video files, sorted
exts = {'.mp4', '.gif'}
videos = sorted([
    f for f in os.listdir(video_dir)
    if os.path.splitext(f)[1].lower() in exts
])

# read prompts (skip empty lines)
with open(prompt_txt, 'r', encoding='utf-8') as f:
    prompts = [line.strip() for line in f if line.strip()]

if len(prompts) != len(videos):
    print(f'Error: {len(prompts)} prompts but {len(videos)} videos found.')
    print(f'Videos: {videos[:5]}...' if len(videos) > 5 else f'Videos: {videos}')
    sys.exit(1)

mapping = {}
for filename, prompt in zip(videos, prompts):
    mapping[filename] = prompt

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)

print(f'Mapped {len(mapping)} videos to prompts.')
" "$VIDEO_DIR" "$PROMPT_TXT" "$PROMPT_JSON"

echo "==> Video folder : $VIDEO_DIR"
echo "==> Prompt file  : $PROMPT_TXT"
echo "==> Prompt JSON  : $PROMPT_JSON"
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
        --prompt_file "$PROMPT_JSON" \
        --output_path "$OUTPUT_DIR" \
        --load_ckpt_from_local True
else
    python -m vbench.launch.evaluate \
        --videos_path "$VIDEO_DIR" \
        --dimension $DIMENSIONS \
        --mode custom_input \
        --prompt_file "$PROMPT_JSON" \
        --output_path "$OUTPUT_DIR" \
        --load_ckpt_from_local True
fi

echo ""
echo "==> Evaluation complete! Results saved to:"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || echo "(no JSON found)"
