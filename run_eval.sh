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
# Results JSON will be saved directly into <video_folder>/

set -e

# ── Parse arguments ──────────────────────────────────────────────
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash run_eval.sh <video_folder> <prompt_txt> [--ngpus N]"
    echo ""
    echo "  video_folder  folder with .mp4/.gif files"
    echo "  prompt_txt    text file, one prompt per line (matched by sorted filename order)"
    exit 1
fi

VIDEO_DIR="$1"
PROMPT_TXT="$2"
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
OUTPUT_DIR="$VIDEO_DIR"

# ── Build prompt JSON ────────────────────────────────────────────
# VBench --prompt_file expects: {"filename": "prompt", ...}
# We match prompts to videos by sorted filename order.
PROMPT_JSON="$OUTPUT_DIR/prompt_map.json"

python3 -c "
import os, json, sys, re

video_dir = sys.argv[1]
prompt_txt = sys.argv[2]
output_json = sys.argv[3]

def first_n_words(text, n=5):
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return tuple(words[:n])

# collect video files
exts = {'.mp4', '.gif'}
videos = [
    f for f in os.listdir(video_dir)
    if os.path.splitext(f)[1].lower() in exts
]

# read prompts (skip empty lines)
with open(prompt_txt, 'r', encoding='utf-8') as f:
    prompts = [line.strip() for line in f if line.strip()]

# build lookup: first 5 words of prompt -> prompt
prompt_lookup = {}
for p in prompts:
    key = first_n_words(p)
    prompt_lookup[key] = p

# match each video filename to a prompt by first 5 words
mapping = {}
unmatched = []
for filename in videos:
    name = os.path.splitext(filename)[0]
    key = first_n_words(name)
    if key in prompt_lookup:
        mapping[filename] = prompt_lookup[key]
    else:
        unmatched.append(filename)

if unmatched:
    print(f'Warning: {len(unmatched)} videos could not be matched:')
    for f in unmatched:
        print(f'  {f}')
    sys.exit(1)

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)

print(f'Matched {len(mapping)} videos to prompts (by first 5 words).')
" "$VIDEO_DIR" "$PROMPT_TXT" "$PROMPT_JSON"

echo "==> Video folder : $VIDEO_DIR"
echo "==> Prompt file  : $PROMPT_TXT"
echo "==> Dimensions   : $DIMENSIONS"
echo "==> GPUs         : $NGPUS"
echo "==> Output       : $VIDEO_DIR (same as video folder)"
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
ls -lh "$VIDEO_DIR"/*.json 2>/dev/null || echo "(no JSON found)"
