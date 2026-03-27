#!/bin/bash
# Full VBench evaluation: runs all 16 dimensions and computes Quality/Semantic/Total.
#
# Usage:
#   bash run_full_eval.sh <video_folder> [--ngpus N]
#
# Video naming convention:
#   <video_folder>/XXXX_seedY.mp4
#   where XXXX is 0-padded index into prompts/all_dimension.txt, Y is seed index.

set -e

# Workaround for cuDNN SDPA compatibility issues on some GPUs
export TORCH_CUDNN_V8_API_DISABLED=1

if [ -z "$1" ]; then
    echo "Usage: bash run_full_eval.sh <video_folder> [--ngpus N]"
    echo ""
    echo "  video_folder  folder with videos named XXXX_seedY.mp4"
    echo "  XXXX = prompt index (0000-0945), matching all_dimension.txt"
    exit 1
fi

VIDEO_DIR="$(cd "$1" && pwd)"
shift

NGPUS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ngpus) NGPUS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FULL_INFO="$SCRIPT_DIR/vbench/VBench_full_info.json"
ALL_PROMPTS="$SCRIPT_DIR/prompts/all_dimension.txt"
OUTPUT_DIR="$VIDEO_DIR/vbench_results"

mkdir -p "$OUTPUT_DIR"

ALL_DIMENSIONS=(
    subject_consistency
    background_consistency
    temporal_flickering
    motion_smoothness
    dynamic_degree
    aesthetic_quality
    imaging_quality
    object_class
    multiple_objects
    human_action
    color
    spatial_relationship
    scene
    appearance_style
    temporal_style
    overall_consistency
)

# Dimensions that require vbench_standard mode (need extra info from prompt)
STANDARD_DIMS="object_class multiple_objects color spatial_relationship scene appearance_style"

echo "============================================================"
echo "  VBench Full Evaluation"
echo "============================================================"
echo "Video folder : $VIDEO_DIR"
echo "Output       : $OUTPUT_DIR"
echo "GPUs         : $NGPUS"
echo "Dimensions   : ${#ALL_DIMENSIONS[@]}"
echo ""

# ── Step 1: Build per-dimension prompt_map.json files ──────────
echo "==> Building per-dimension prompt maps..."

python3 -c "
import json, os, sys, glob
from collections import defaultdict

video_dir = sys.argv[1]
full_info_path = sys.argv[2]
all_prompts_path = sys.argv[3]
output_dir = sys.argv[4]

# Dimensions that need vbench_standard mode (prompt-named symlinks)
STANDARD_DIMS = {'object_class', 'multiple_objects', 'color', 'spatial_relationship', 'scene', 'appearance_style'}

# Read all_dimension.txt to get prompt list (index -> prompt)
with open(all_prompts_path, 'r', encoding='utf-8') as f:
    all_prompts = [line.strip() for line in f if line.strip()]

# Read VBench_full_info.json
with open(full_info_path, 'r', encoding='utf-8') as f:
    full_info = json.load(f)

# Build: prompt_text -> list of dimensions
prompt_to_dims = {}
for item in full_info:
    prompt_to_dims[item['prompt_en']] = item['dimension']

# Scan video files
video_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
if not video_files:
    video_files = sorted(glob.glob(os.path.join(video_dir, '*.gif')))

postfix = os.path.splitext(video_files[0])[1] if video_files else '.mp4'

# Parse video filenames: XXXX_seedY.ext -> (index, seed)
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
            print(f'  Warning: cannot parse {basename}, skipping')
    else:
        idx_str = name.split('_')[0]
        try:
            idx = int(idx_str)
            index_to_files[idx].append((basename, len(index_to_files.get(idx, []))))
        except ValueError:
            print(f'  Warning: cannot parse index from {basename}, skipping')

print(f'  Found {len(video_files)} video files, {len(index_to_files)} unique prompts')

# Build per-dimension maps
dim_maps = defaultdict(dict)  # dim -> {filename: prompt}
for idx, file_seed_list in index_to_files.items():
    if idx >= len(all_prompts):
        continue
    prompt = all_prompts[idx]
    dims = prompt_to_dims.get(prompt, [])
    if not dims:
        print(f'  Warning: prompt {idx} not found in VBench_full_info.json: {prompt[:60]}...')
        continue
    for dim in dims:
        for fn, seed in file_seed_list:
            dim_maps[dim][(fn, seed)] = prompt

# Write per-dimension JSON files and create symlink subdirectories
for dim, mapping in sorted(dim_maps.items()):
    dim_video_dir = os.path.join(output_dir, f'videos_{dim}')
    os.makedirs(dim_video_dir, exist_ok=True)

    if dim in STANDARD_DIMS:
        # vbench_standard mode: create symlinks named {prompt}-{i}.mp4
        for (fn, seed), prompt in mapping.items():
            standard_name = f'{prompt}-{seed}{postfix}'
            src = os.path.join(video_dir, fn)
            dst = os.path.join(dim_video_dir, standard_name)
            if not os.path.exists(dst):
                os.symlink(src, dst)
    else:
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

    print(f'  {dim}: {len(mapping)} videos {\"(standard mode)\" if dim in STANDARD_DIMS else \"(custom mode)\"}')
" "$VIDEO_DIR" "$FULL_INFO" "$ALL_PROMPTS" "$OUTPUT_DIR"

echo ""

# ── Step 2: Run evaluation per dimension ──────────────────────
cd "$SCRIPT_DIR"

for DIM in "${ALL_DIMENSIONS[@]}"; do
    DIM_VIDEO_DIR="$OUTPUT_DIR/videos_${DIM}"
    if [ ! -d "$DIM_VIDEO_DIR" ]; then
        echo "==> SKIP $DIM (no video dir found)"
        continue
    fi

    # Determine mode: standard for dims needing extra info, custom for others
    IS_STANDARD=false
    for SD in $STANDARD_DIMS; do
        if [ "$DIM" = "$SD" ]; then
            IS_STANDARD=true
            break
        fi
    done

    echo "========================================"
    if $IS_STANDARD; then
        echo "  Evaluating: $DIM (vbench_standard mode)"
    else
        echo "  Evaluating: $DIM (custom_input mode)"
    fi
    echo "========================================"

    if $IS_STANDARD; then
        # vbench_standard mode: videos named {prompt}-{i}.mp4, no prompt_file
        EVAL_CMD="python -m vbench.launch.evaluate \
            --videos_path $DIM_VIDEO_DIR \
            --dimension $DIM \
            --mode vbench_standard \
            --output_path $OUTPUT_DIR \
            --load_ckpt_from_local True"
    else
        PROMPT_MAP="$OUTPUT_DIR/prompt_map_${DIM}.json"
        if [ ! -f "$PROMPT_MAP" ]; then
            echo "==> SKIP $DIM (no prompt map found)"
            continue
        fi
        EVAL_CMD="python -m vbench.launch.evaluate \
            --videos_path $DIM_VIDEO_DIR \
            --dimension $DIM \
            --mode custom_input \
            --prompt_file $PROMPT_MAP \
            --output_path $OUTPUT_DIR \
            --load_ckpt_from_local True"
    fi

    if [ "$NGPUS" -gt 1 ]; then
        EVAL_CMD="${EVAL_CMD/python -m/torchrun --nproc_per_node=$NGPUS -m}"
    fi

    eval $EVAL_CMD || echo "  WARNING: $DIM evaluation failed, continuing..."
    echo ""
done

# ── Step 3: Collect results and compute final scores ──────────
echo "========================================"
echo "  Computing final scores"
echo "========================================"

python3 -c "
import json, os, sys, glob

output_dir = sys.argv[1]

DIM_WEIGHT = {
    'subject consistency':1, 'background consistency':1,
    'temporal flickering':1, 'motion smoothness':1,
    'aesthetic quality':1, 'imaging quality':1,
    'dynamic degree':0.5, 'object class':1,
    'multiple objects':1, 'human action':1,
    'color':1, 'spatial relationship':1,
    'scene':1, 'appearance style':1,
    'temporal style':1, 'overall consistency':1
}

NORMALIZE_DIC = {
    'subject consistency': {'Min': 0.1462, 'Max': 1.0},
    'background consistency': {'Min': 0.2615, 'Max': 1.0},
    'temporal flickering': {'Min': 0.6293, 'Max': 1.0},
    'motion smoothness': {'Min': 0.706, 'Max': 0.9975},
    'dynamic degree': {'Min': 0.0, 'Max': 1.0},
    'aesthetic quality': {'Min': 0.0, 'Max': 1.0},
    'imaging quality': {'Min': 0.0, 'Max': 1.0},
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

QUALITY_LIST = [
    'subject consistency', 'background consistency', 'temporal flickering',
    'motion smoothness', 'aesthetic quality', 'imaging quality', 'dynamic degree'
]
SEMANTIC_LIST = [
    'object class', 'multiple objects', 'human action', 'color',
    'spatial relationship', 'scene', 'appearance style', 'temporal style',
    'overall consistency'
]

# Collect raw scores from all result JSON files
raw_scores = {}
result_files = glob.glob(os.path.join(output_dir, 'results_*_eval_results.json'))
for rf in result_files:
    with open(rf) as f:
        data = json.load(f)
    for key, val in data.items():
        dim_name = key.replace('_', ' ')
        if isinstance(val, list) and len(val) > 0:
            raw_scores[dim_name] = val[0]
        elif isinstance(val, (int, float)):
            raw_scores[dim_name] = val

# Print per-dimension results
print()
print('=' * 60)
print(f'  {\"Dimension\":<30s} {\"Score (x100)\":>12s}  Category')
print('=' * 60)

all_dims = QUALITY_LIST + SEMANTIC_LIST
for dim in all_dims:
    tag = 'Quality' if dim in QUALITY_LIST else 'Semantic'
    if dim in raw_scores:
        print(f'  {dim:<30s} {raw_scores[dim]*100:>10.2f}   {tag}')
    else:
        print(f'  {dim:<30s} {\"MISSING\":>10s}   {tag}')

# Compute normalized scores
normalized = {}
for dim in all_dims:
    if dim not in raw_scores:
        continue
    mn = NORMALIZE_DIC[dim]['Min']
    mx = NORMALIZE_DIC[dim]['Max']
    norm = (raw_scores[dim] - mn) / (mx - mn)
    normalized[dim] = norm * DIM_WEIGHT[dim]

# Quality score
q_dims = [d for d in QUALITY_LIST if d in normalized]
quality = sum(normalized[d] for d in q_dims) / sum(DIM_WEIGHT[d] for d in q_dims) if q_dims else 0

# Semantic score
s_dims = [d for d in SEMANTIC_LIST if d in normalized]
semantic = sum(normalized[d] for d in s_dims) / sum(DIM_WEIGHT[d] for d in s_dims) if s_dims else 0

# Total score
total = (4 * quality + 1 * semantic) / 5

print()
print('=' * 60)
print(f'  Quality  : {quality*100:.2f}  ({len(q_dims)}/{len(QUALITY_LIST)} dims)')
print(f'  Semantic : {semantic*100:.2f}  ({len(s_dims)}/{len(SEMANTIC_LIST)} dims)')
print(f'  Total    : {total*100:.2f}')
print('=' * 60)

# Save summary
summary = {
    'per_dimension': {k: round(v*100, 2) for k, v in raw_scores.items()},
    'quality_score': round(quality*100, 2),
    'semantic_score': round(semantic*100, 2),
    'total_score': round(total*100, 2),
}
summary_path = os.path.join(output_dir, 'final_scores.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f'\nSaved to {summary_path}')
" "$OUTPUT_DIR"

echo ""
echo "==> Full evaluation complete!"
