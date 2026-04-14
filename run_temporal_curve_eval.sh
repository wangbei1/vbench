#!/bin/bash
# Temporal quality curve evaluation.
#
# For each method's video folder (containing 100s videos), truncate to
# different lengths {5, 15, 30, 45, 60, 80, 100}s and run VBench Long
# evaluation. Outputs JSON files for each (method, length) combination.
#
# Usage:
#   bash run_temporal_curve_eval.sh <output_root> <method_name> <video_folder>
#
# Example:
#   bash run_temporal_curve_eval.sh /path/to/curve_results ours /path/to/ours/100s_videos
#   bash run_temporal_curve_eval.sh /path/to/curve_results longlive /path/to/longlive/100s_videos
#   bash run_temporal_curve_eval.sh /path/to/curve_results reward_forcing /path/to/rf/100s_videos
#
# Then run plot_temporal_curve.py to generate the figure.

set -e
export TORCH_CUDNN_V8_API_DISABLED=1
export MASTER_PORT=$(shuf -i 29500-39999 -n 1)

if [ $# -lt 3 ]; then
    echo "Usage: bash run_temporal_curve_eval.sh <output_root> <method_name> <video_folder>"
    exit 1
fi

OUTPUT_ROOT="$1"
METHOD="$2"
VIDEO_DIR="$(cd "$3" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LENGTHS=(5 15 30 45 60 80 100)
DIMS=(subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality)

mkdir -p "$OUTPUT_ROOT/$METHOD"

echo "============================================================"
echo "  Temporal Curve Evaluation"
echo "============================================================"
echo "Method      : $METHOD"
echo "Source      : $VIDEO_DIR"
echo "Output root : $OUTPUT_ROOT/$METHOD"
echo "Lengths     : ${LENGTHS[*]}"
echo ""

for LEN in "${LENGTHS[@]}"; do
    TRUNC_DIR="$OUTPUT_ROOT/$METHOD/videos_${LEN}s"
    RESULT_DIR="$OUTPUT_ROOT/$METHOD/results_${LEN}s"
    SUMMARY_FILE="$OUTPUT_ROOT/$METHOD/scores_${LEN}s.json"

    if [ -f "$SUMMARY_FILE" ]; then
        echo "==> SKIP ${LEN}s (already done): $SUMMARY_FILE"
        continue
    fi

    echo "============================================================"
    echo "  Truncating to ${LEN}s"
    echo "============================================================"
    mkdir -p "$TRUNC_DIR"

    # Truncate all videos in parallel using ffmpeg stream-copy
    for f in "$VIDEO_DIR"/*.mp4; do
        name=$(basename "$f")
        if [ ! -f "$TRUNC_DIR/$name" ]; then
            ffmpeg -y -loglevel error -i "$f" -t "$LEN" -c copy "$TRUNC_DIR/$name" &
            # limit parallelism
            if (( $(jobs -r | wc -l) >= 16 )); then
                wait -n
            fi
        fi
    done
    wait
    echo "Truncated $(ls "$TRUNC_DIR"/*.mp4 | wc -l) videos to ${LEN}s"

    # Clean up old split_clip if exists
    rm -rf "$TRUNC_DIR/split_clip"

    mkdir -p "$RESULT_DIR"

    echo "============================================================"
    echo "  Evaluating ${METHOD} @ ${LEN}s"
    echo "============================================================"
    cd "$SCRIPT_DIR"

    for DIM in "${DIMS[@]}"; do
        echo "  -- $DIM"
        python vbench2_beta_long/eval_long.py \
            --videos_path "$TRUNC_DIR" \
            --dimension "$DIM" \
            --mode long_custom_input \
            --output_path "$RESULT_DIR" \
            --load_ckpt_from_local True \
            --dev_flag \
        || echo "  WARNING: $DIM failed at ${LEN}s"
    done

    # Aggregate scores into a summary JSON
    python3 -c "
import json, os, glob

result_dir = '$RESULT_DIR'
out_file = '$SUMMARY_FILE'

scores = {}
for rf in glob.glob(os.path.join(result_dir, 'results_*_eval_results.json')):
    with open(rf) as f:
        data = json.load(f)
    for key, val in data.items():
        dim = key.replace('_', ' ')
        if isinstance(val, list) and len(val) > 0:
            scores[dim] = val[0]
        elif isinstance(val, (int, float)):
            scores[dim] = val

with open(out_file, 'w') as f:
    json.dump(scores, f, indent=2)
print(f'Saved: {out_file} -> {scores}')
"

    # Optional: clean up truncated videos to save disk
    # rm -rf "$TRUNC_DIR"
done

echo ""
echo "==> Temporal curve evaluation complete for $METHOD"
echo "==> Run plot_temporal_curve.py to generate the figure"
