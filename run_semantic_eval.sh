#!/bin/bash
# Run Semantic dimensions of VBench evaluation.
#
# Usage:
#   bash run_semantic_eval.sh <video_folder> [dimension ...]
#
# Examples:
#   bash run_semantic_eval.sh /path/to/videos                   # all 9 semantic dims
#   bash run_semantic_eval.sh /path/to/videos human_action      # only human_action
#   bash run_semantic_eval.sh /path/to/videos human_action scene color
#
# Video naming convention:
#   <video_folder>/XXXX_seedY.mp4

set -e
export TORCH_CUDNN_V8_API_DISABLED=1

if [ -z "$1" ]; then
    echo "Usage: bash run_semantic_eval.sh <video_folder> [dimension ...]"
    exit 1
fi

VIDEO_DIR="$(cd "$1" && pwd)"
shift
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FULL_INFO="$SCRIPT_DIR/vbench/VBench_full_info.json"
ALL_PROMPTS="$SCRIPT_DIR/prompts/all_dimension.txt"
OUTPUT_DIR="$VIDEO_DIR/vbench_results"

mkdir -p "$OUTPUT_DIR"

# All semantic dimensions
ALL_CUSTOM_DIMS=(human_action temporal_style overall_consistency)
ALL_STANDARD_DIMS=(object_class multiple_objects color spatial_relationship scene appearance_style)

# Filter by user-specified dimensions (if any)
if [ $# -gt 0 ]; then
    CUSTOM_DIMS=()
    STANDARD_DIMS=()
    for dim in "$@"; do
        for d in "${ALL_CUSTOM_DIMS[@]}"; do
            if [ "$dim" = "$d" ]; then CUSTOM_DIMS+=("$dim"); fi
        done
        for d in "${ALL_STANDARD_DIMS[@]}"; do
            if [ "$dim" = "$d" ]; then STANDARD_DIMS+=("$dim"); fi
        done
    done
else
    CUSTOM_DIMS=("${ALL_CUSTOM_DIMS[@]}")
    STANDARD_DIMS=("${ALL_STANDARD_DIMS[@]}")
fi

ALL_SEMANTIC=("${CUSTOM_DIMS[@]}" "${STANDARD_DIMS[@]}")

ALL_SEMANTIC=("${CUSTOM_DIMS[@]}" "${STANDARD_DIMS[@]}")

echo "============================================================"
echo "  VBench Semantic Evaluation (${#ALL_SEMANTIC[@]} dimensions)"
echo "============================================================"
echo "Video folder : $VIDEO_DIR"
echo "Output       : $OUTPUT_DIR"
echo "Dimensions   : ${ALL_SEMANTIC[*]}"
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

STANDARD_DIMS = {'object_class', 'multiple_objects', 'color', 'spatial_relationship', 'scene', 'appearance_style'}
SEMANTIC_DIMS = STANDARD_DIMS | {'human_action', 'temporal_style', 'overall_consistency'}

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
        if dim not in SEMANTIC_DIMS:
            continue
        for fn, seed in file_seed_list:
            dim_maps[dim][(fn, seed)] = prompt

for dim, mapping in sorted(dim_maps.items()):
    dim_video_dir = os.path.join(output_dir, f'videos_{dim}')
    os.makedirs(dim_video_dir, exist_ok=True)

    if dim in STANDARD_DIMS:
        for (fn, seed), prompt in mapping.items():
            standard_name = f'{prompt}-{seed}{postfix}'
            src = os.path.join(video_dir, fn)
            dst = os.path.join(dim_video_dir, standard_name)
            if not os.path.exists(dst):
                os.symlink(src, dst)
    else:
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

# ── Helper: print current scores ─────────────────────────────
print_scores() {
    python3 -c "
import json, os, sys, glob

output_dir = sys.argv[1]

SEMANTIC_LIST = [
    'object class', 'multiple objects', 'human action', 'color',
    'spatial relationship', 'scene', 'appearance style', 'temporal style',
    'overall consistency'
]
NORMALIZE_DIC = {
    'object class': {'Min': 0.0, 'Max': 1.0},
    'multiple objects': {'Min': 0.0, 'Max': 1.0},
    'human action': {'Min': 0.0, 'Max': 1.0},
    'color': {'Min': 0.0, 'Max': 1.0},
    'spatial relationship': {'Min': 0.0, 'Max': 1.0},
    'scene': {'Min': 0.0, 'Max': 0.8222},
    'appearance style': {'Min': 0.0009, 'Max': 0.2855},
    'temporal style': {'Min': 0.0, 'Max': 0.364},
    'overall consistency': {'Min': 0.0, 'Max': 0.364}
}

raw_scores = {}
for rf in glob.glob(os.path.join(output_dir, 'results_*_eval_results.json')):
    with open(rf) as f:
        data = json.load(f)
    for key, val in data.items():
        dim_name = key.replace('_', ' ')
        if dim_name in SEMANTIC_LIST:
            if isinstance(val, list) and len(val) > 0:
                raw_scores[dim_name] = val[0]
            elif isinstance(val, (int, float)):
                raw_scores[dim_name] = val

if not raw_scores:
    return

normalized = {}
for dim in SEMANTIC_LIST:
    if dim not in raw_scores:
        continue
    mn = NORMALIZE_DIC[dim]['Min']
    mx = NORMALIZE_DIC[dim]['Max']
    normalized[dim] = (raw_scores[dim] - mn) / (mx - mn)

s_dims = [d for d in SEMANTIC_LIST if d in normalized]
semantic = sum(normalized[d] for d in s_dims) / len(s_dims) if s_dims else 0

print()
print('-' * 55)
for dim in SEMANTIC_LIST:
    if dim in raw_scores:
        print(f'  {dim:<25s} {raw_scores[dim]*100:>10.2f}')
print(f'  Semantic={semantic*100:.2f} ({len(s_dims)}/9)')
print('-' * 55)
" "$OUTPUT_DIR"
}

# ── Step 2: Run evaluation ────────────────────────────────────
cd "$SCRIPT_DIR"

# Custom input dimensions
for DIM in "${CUSTOM_DIMS[@]}"; do
    DIM_VIDEO_DIR="$OUTPUT_DIR/videos_${DIM}"
    PROMPT_MAP="$OUTPUT_DIR/prompt_map_${DIM}.json"
    if [ ! -d "$DIM_VIDEO_DIR" ] || [ ! -f "$PROMPT_MAP" ]; then
        echo "==> SKIP $DIM"
        continue
    fi
    echo "========================================"
    echo "  Evaluating: $DIM (custom_input)"
    echo "========================================"
    python -m vbench.launch.evaluate \
        --videos_path "$DIM_VIDEO_DIR" \
        --dimension "$DIM" \
        --mode custom_input \
        --prompt_file "$PROMPT_MAP" \
        --output_path "$OUTPUT_DIR" \
        --load_ckpt_from_local True \
    || echo "  WARNING: $DIM failed"
    print_scores
    echo ""
done

# Standard mode dimensions
for DIM in "${STANDARD_DIMS[@]}"; do
    DIM_VIDEO_DIR="$OUTPUT_DIR/videos_${DIM}"
    if [ ! -d "$DIM_VIDEO_DIR" ]; then
        echo "==> SKIP $DIM"
        continue
    fi
    echo "========================================"
    echo "  Evaluating: $DIM (vbench_standard)"
    echo "========================================"
    python -m vbench.launch.evaluate \
        --videos_path "$DIM_VIDEO_DIR" \
        --dimension "$DIM" \
        --mode vbench_standard \
        --output_path "$OUTPUT_DIR" \
        --load_ckpt_from_local True \
    || echo "  WARNING: $DIM failed"
    print_scores
    echo ""
done

# ── Step 3: Collect and print Semantic scores ─────────────────
echo "========================================"
echo "  Semantic Scores"
echo "========================================"

python3 -c "
import json, os, sys, glob

output_dir = sys.argv[1]

SEMANTIC_LIST = [
    'object class', 'multiple objects', 'human action', 'color',
    'spatial relationship', 'scene', 'appearance style', 'temporal style',
    'overall consistency'
]
DIM_WEIGHT = {d: 1 for d in SEMANTIC_LIST}
NORMALIZE_DIC = {
    'object class': {'Min': 0.0, 'Max': 1.0},
    'multiple objects': {'Min': 0.0, 'Max': 1.0},
    'human action': {'Min': 0.0, 'Max': 1.0},
    'color': {'Min': 0.0, 'Max': 1.0},
    'spatial relationship': {'Min': 0.0, 'Max': 1.0},
    'scene': {'Min': 0.0, 'Max': 0.8222},
    'appearance style': {'Min': 0.0009, 'Max': 0.2855},
    'temporal style': {'Min': 0.0, 'Max': 0.364},
    'overall consistency': {'Min': 0.0, 'Max': 0.364}
}

raw_scores = {}
for rf in glob.glob(os.path.join(output_dir, 'results_*_eval_results.json')):
    with open(rf) as f:
        data = json.load(f)
    for key, val in data.items():
        dim_name = key.replace('_', ' ')
        if dim_name in SEMANTIC_LIST:
            if isinstance(val, list) and len(val) > 0:
                raw_scores[dim_name] = val[0]
            elif isinstance(val, (int, float)):
                raw_scores[dim_name] = val

print()
print('=' * 55)
print(f'  {\"Dimension\":<25s} {\"Score (x100)\":>12s}')
print('=' * 55)
for dim in SEMANTIC_LIST:
    if dim in raw_scores:
        print(f'  {dim:<25s} {raw_scores[dim]*100:>10.2f}')
    else:
        print(f'  {dim:<25s} {\"MISSING\":>10s}')

normalized = {}
for dim in SEMANTIC_LIST:
    if dim not in raw_scores:
        continue
    mn = NORMALIZE_DIC[dim]['Min']
    mx = NORMALIZE_DIC[dim]['Max']
    normalized[dim] = (raw_scores[dim] - mn) / (mx - mn)

s_dims = [d for d in SEMANTIC_LIST if d in normalized]
semantic = sum(normalized[d] for d in s_dims) / len(s_dims) if s_dims else 0

print()
print(f'  Semantic Score: {semantic*100:.2f}  ({len(s_dims)}/9 dims)')
print('=' * 55)
" "$OUTPUT_DIR"

echo ""
echo "==> Semantic evaluation complete!"
