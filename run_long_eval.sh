#!/bin/bash
# VBench Long video evaluation.
#
# Reproduces the metrics from Table 2:
#   subject_consistency, background_consistency, motion_smoothness,
#   dynamic_degree, aesthetic_quality, imaging_quality
#   + Total score (normalized & weighted)
#   + Drift (std of imaging_quality across segments)
#
# Usage:
#   bash run_long_eval.sh <video_folder> [dimension ...]
#
# Examples:
#   bash run_long_eval.sh /path/to/videos                        # all 6 quality dims
#   bash run_long_eval.sh /path/to/videos imaging_quality        # only imaging_quality
#   bash run_long_eval.sh /path/to/videos dynamic_degree aesthetic_quality
#
# Video naming convention:
#   <video_folder>/XXXX_seedY.mp4  (60-second videos)
#   Videos are automatically split into 2-second clips.

set -e
export TORCH_CUDNN_V8_API_DISABLED=1

if [ -z "$1" ]; then
    echo "Usage: bash run_long_eval.sh <video_folder> [dimension ...]"
    exit 1
fi

VIDEO_DIR="$(cd "$1" && pwd)"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$VIDEO_DIR/vbench_long_results"

mkdir -p "$OUTPUT_DIR"

# Default: all 6 quality dimensions from Table 2
ALL_DIMS=(subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality)

if [ $# -gt 0 ]; then
    DIMS=("$@")
else
    DIMS=("${ALL_DIMS[@]}")
fi

echo "============================================================"
echo "  VBench Long Video Evaluation"
echo "============================================================"
echo "Video folder : $VIDEO_DIR"
echo "Output       : $OUTPUT_DIR"
echo "Dimensions   : ${DIMS[*]}"
echo ""

# ── Run evaluation per dimension ──────────────────────────────
cd "$SCRIPT_DIR"

for DIM in "${DIMS[@]}"; do
    echo "========================================"
    echo "  Evaluating: $DIM"
    echo "========================================"
    python vbench2_beta_long/eval_long.py \
        --videos_path "$VIDEO_DIR" \
        --dimension "$DIM" \
        --mode long_custom_input \
        --output_path "$OUTPUT_DIR" \
        --load_ckpt_from_local True \
        --dev_flag \
    || echo "  WARNING: $DIM failed"
    echo ""
done

# ── Collect results and compute scores ────────────────────────
echo "========================================"
echo "  Computing scores"
echo "========================================"

python3 -c "
import json, os, sys, glob
import numpy as np

output_dir = sys.argv[1]

# VBench standard normalization coefficients
NORMALIZE_DIC = {
    'subject consistency': {'Min': 0.1462, 'Max': 1.0},
    'background consistency': {'Min': 0.2615, 'Max': 1.0},
    'motion smoothness': {'Min': 0.706, 'Max': 0.9975},
    'dynamic degree': {'Min': 0.0, 'Max': 1.0},
    'aesthetic quality': {'Min': 0.0, 'Max': 1.0},
    'imaging quality': {'Min': 0.0, 'Max': 1.0},
}
DIM_WEIGHT = {d: (0.5 if d == 'dynamic degree' else 1.0) for d in NORMALIZE_DIC}

# Collect raw scores
raw_scores = {}
per_video_iq = {}  # for drift calculation

result_files = glob.glob(os.path.join(output_dir, 'results_*_eval_results.json'))
for rf in result_files:
    with open(rf) as f:
        data = json.load(f)
    for key, val in data.items():
        dim_name = key.replace('_', ' ')
        if dim_name not in NORMALIZE_DIC:
            continue
        if isinstance(val, list):
            if len(val) >= 3:
                # [overall_score, detailed_results, per_video_averages]
                raw_scores[dim_name] = val[0]
                if dim_name == 'imaging quality':
                    # Collect per-clip scores for drift
                    per_video_iq = val[1] if len(val) > 1 else []
            elif len(val) > 0:
                raw_scores[dim_name] = val[0]
        elif isinstance(val, (int, float)):
            raw_scores[dim_name] = val

# Print per-dimension results
print()
print('=' * 55)
print(f'  {\"Dimension\":<25s} {\"Score (x100)\":>12s}')
print('=' * 55)
for dim in NORMALIZE_DIC:
    if dim in raw_scores:
        print(f'  {dim:<25s} {raw_scores[dim]*100:>10.2f}')
    else:
        print(f'  {dim:<25s} {\"MISSING\":>10s}')

# Compute normalized total score
normalized = {}
for dim in NORMALIZE_DIC:
    if dim not in raw_scores:
        continue
    mn = NORMALIZE_DIC[dim]['Min']
    mx = NORMALIZE_DIC[dim]['Max']
    normalized[dim] = (raw_scores[dim] - mn) / (mx - mn) * DIM_WEIGHT[dim]

avail = [d for d in NORMALIZE_DIC if d in normalized]
total = sum(normalized[d] for d in avail) / sum(DIM_WEIGHT[d] for d in avail) if avail else 0

print()
print(f'  Total Score: {total*100:.2f}  ({len(avail)}/{len(NORMALIZE_DIC)} dims)')

# Compute drift (std of imaging quality across segments per video)
if per_video_iq and isinstance(per_video_iq, list):
    # per_video_iq is detailed_results: list of {video_path, video_results}
    from collections import defaultdict
    video_clip_scores = defaultdict(list)
    for item in per_video_iq:
        if not isinstance(item, dict) or 'video_path' not in item:
            continue
        clip_path = item['video_path']
        if 'split_clip' in clip_path:
            video_name = os.path.basename(os.path.dirname(clip_path))
        else:
            video_name = os.path.basename(clip_path)
        video_clip_scores[video_name].append(item['video_results'])

    if video_clip_scores:
        per_video_std = [np.std(scores) for scores in video_clip_scores.values() if len(scores) > 1]
        if per_video_std:
            drift = np.mean(per_video_std)
            print(f'  Drift (IQ std): {drift:.3f}')

print('=' * 55)

# Save summary
summary = {
    'per_dimension': {k: round(v*100, 2) for k, v in raw_scores.items()},
    'total_score': round(total*100, 2),
}
summary_path = os.path.join(output_dir, 'long_eval_scores.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f'\nSaved to {summary_path}')
" "$OUTPUT_DIR"

echo ""
echo "==> Long video evaluation complete!"
