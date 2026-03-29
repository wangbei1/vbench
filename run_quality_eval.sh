#!/bin/bash
# Run Quality dimensions of VBench evaluation.
#
# Usage:
#   bash run_quality_eval.sh <video_folder> [dimension ...]
#
# Examples:
#   bash run_quality_eval.sh /path/to/videos                              # all 7 quality dims
#   bash run_quality_eval.sh /path/to/videos background_consistency       # only one
#   bash run_quality_eval.sh /path/to/videos aesthetic_quality imaging_quality
#
# Video naming convention:
#   <video_folder>/XXXX_seedY.mp4

set -e
export TORCH_CUDNN_V8_API_DISABLED=1

if [ -z "$1" ]; then
    echo "Usage: bash run_quality_eval.sh <video_folder> [dimension ...]"
    exit 1
fi

VIDEO_DIR="$(cd "$1" && pwd)"
shift
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FULL_INFO="$SCRIPT_DIR/vbench/VBench_full_info.json"
ALL_PROMPTS="$SCRIPT_DIR/prompts/all_dimension.txt"
OUTPUT_DIR="$VIDEO_DIR/vbench_results"

mkdir -p "$OUTPUT_DIR"

# Quality dimensions: all use custom_input mode with prompt_map
ALL_QUALITY_DIMS=(subject_consistency background_consistency temporal_flickering motion_smoothness dynamic_degree aesthetic_quality imaging_quality)

# Filter by user-specified dimensions (if any)
if [ $# -gt 0 ]; then
    DIMS=()
    for dim in "$@"; do
        for d in "${ALL_QUALITY_DIMS[@]}"; do
            if [ "$dim" = "$d" ]; then DIMS+=("$dim"); fi
        done
    done
else
    DIMS=("${ALL_QUALITY_DIMS[@]}")
fi

echo "============================================================"
echo "  VBench Quality Evaluation (${#DIMS[@]} dimensions)"
echo "============================================================"
echo "Video folder : $VIDEO_DIR"
echo "Output       : $OUTPUT_DIR"
echo "Dimensions   : ${DIMS[*]}"
echo ""

# ── Step 1: Build per-dimension video dirs and prompt maps ─────
echo "==> Preparing per-dimension data..."

python3 -c "
import json, os, sys, glob
from collections import defaultdict

video_dir = sys.argv[1]
full_info_path = sys.argv[2]
all_prompts_path = sys.argv[3]
output_dir = sys.argv[4]

QUALITY_DIMS = {'subject_consistency', 'background_consistency', 'temporal_flickering',
                'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'}

with open(all_prompts_path, 'r', encoding='utf-8') as f:
    all_prompts = [line.strip() for line in f if line.strip()]

with open(full_info_path, 'r', encoding='utf-8') as f:
    full_info = json.load(f)

prompt_to_dims = {}
for item in full_info:
    prompt_to_dims[item['prompt_en']] = item['dimension']

video_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
if not video_files:
    video_files = sorted(glob.glob(os.path.join(video_dir, '*.gif')))

postfix = os.path.splitext(video_files[0])[1] if video_files else '.mp4'

index_to_files = defaultdict(list)
for vf in video_files:
    basename = os.path.basename(vf)
    name = os.path.splitext(basename)[0]
    parts = name.split('_seed')
    if len(parts) == 2:
        try:
            idx = int(parts[0])
            seed = int(parts[1])
            index_to_files[idx].append((basename, seed))
        except ValueError:
            pass

dim_maps = defaultdict(dict)
for idx, file_seed_list in index_to_files.items():
    if idx >= len(all_prompts):
        continue
    prompt = all_prompts[idx]
    dims = prompt_to_dims.get(prompt, [])
    for dim in dims:
        if dim not in QUALITY_DIMS:
            continue
        for fn, seed in file_seed_list:
            dim_maps[dim][(fn, seed)] = prompt

for dim, mapping in sorted(dim_maps.items()):
    dim_video_dir = os.path.join(output_dir, f'videos_{dim}')
    os.makedirs(dim_video_dir, exist_ok=True)

    # custom_input mode: keep original names + prompt_map.json
    prompt_map = {}
    for (fn, seed), prompt in mapping.items():
        prompt_map[fn] = prompt
        src = os.path.join(video_dir, fn)
        dst = os.path.join(dim_video_dir, fn)
        if not os.path.exists(dst):
            os.symlink(src, dst)
    out_path = os.path.join(output_dir, f'prompt_map_{dim}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_map, f, indent=2, ensure_ascii=False)

    print(f'  {dim}: {len(mapping)} videos')
" "$VIDEO_DIR" "$FULL_INFO" "$ALL_PROMPTS" "$OUTPUT_DIR"

echo ""

# ── Step 2: Run evaluation ────────────────────────────────────
cd "$SCRIPT_DIR"

for DIM in "${DIMS[@]}"; do
    DIM_VIDEO_DIR="$OUTPUT_DIR/videos_${DIM}"
    PROMPT_MAP="$OUTPUT_DIR/prompt_map_${DIM}.json"
    if [ ! -d "$DIM_VIDEO_DIR" ] || [ ! -f "$PROMPT_MAP" ]; then
        echo "==> SKIP $DIM (no video dir or prompt map found)"
        continue
    fi
    echo "========================================"
    echo "  Evaluating: $DIM"
    echo "========================================"
    python -m vbench.launch.evaluate \
        --videos_path "$DIM_VIDEO_DIR" \
        --dimension "$DIM" \
        --mode custom_input \
        --prompt_file "$PROMPT_MAP" \
        --output_path "$OUTPUT_DIR" \
        --load_ckpt_from_local True \
    || echo "  WARNING: $DIM failed"
    echo ""
done

# ── Step 3: Collect and print Quality scores ──────────────────
echo "========================================"
echo "  Quality Scores"
echo "========================================"

python3 -c "
import json, os, sys, glob

output_dir = sys.argv[1]

QUALITY_LIST = [
    'subject consistency', 'background consistency', 'temporal flickering',
    'motion smoothness', 'dynamic degree', 'aesthetic quality', 'imaging quality'
]
DIM_WEIGHT = {d: (0.5 if d == 'dynamic degree' else 1.0) for d in QUALITY_LIST}
NORMALIZE_DIC = {
    'subject consistency': {'Min': 0.1462, 'Max': 1.0},
    'background consistency': {'Min': 0.2615, 'Max': 1.0},
    'temporal flickering': {'Min': 0.6293, 'Max': 1.0},
    'motion smoothness': {'Min': 0.706, 'Max': 0.9975},
    'dynamic degree': {'Min': 0.0, 'Max': 1.0},
    'aesthetic quality': {'Min': 0.0, 'Max': 1.0},
    'imaging quality': {'Min': 0.0, 'Max': 1.0},
}

raw_scores = {}
for rf in glob.glob(os.path.join(output_dir, 'results_*_eval_results.json')):
    with open(rf) as f:
        data = json.load(f)
    for key, val in data.items():
        dim_name = key.replace('_', ' ')
        if dim_name in QUALITY_LIST:
            if isinstance(val, list) and len(val) > 0:
                raw_scores[dim_name] = val[0]
            elif isinstance(val, (int, float)):
                raw_scores[dim_name] = val

print()
print('=' * 55)
print(f'  {\"Dimension\":<25s} {\"Score (x100)\":>12s}')
print('=' * 55)
for dim in QUALITY_LIST:
    if dim in raw_scores:
        print(f'  {dim:<25s} {raw_scores[dim]*100:>10.2f}')
    else:
        print(f'  {dim:<25s} {\"MISSING\":>10s}')

normalized = {}
for dim in QUALITY_LIST:
    if dim not in raw_scores:
        continue
    mn = NORMALIZE_DIC[dim]['Min']
    mx = NORMALIZE_DIC[dim]['Max']
    normalized[dim] = (raw_scores[dim] - mn) / (mx - mn) * DIM_WEIGHT[dim]

q_dims = [d for d in QUALITY_LIST if d in normalized]
quality = sum(normalized[d] for d in q_dims) / sum(DIM_WEIGHT[d] for d in q_dims) if q_dims else 0

print()
print(f'  Quality Score: {quality*100:.2f}  ({len(q_dims)}/7 dims)')
print('=' * 55)
" "$OUTPUT_DIR"

echo ""
echo "==> Quality evaluation complete!"
