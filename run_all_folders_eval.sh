#!/bin/bash
# Evaluate every video folder (focalreward & Reward-Forcing at 10/30/60/120/180s)
# and write a single summary TXT.
#
# Long video eval is NOT multi-GPU aware internally — vbench2_beta_long has
# zero uses of distribute_list_to_rank / init_process_group. Confirmed:
#   grep -rn distribute_list_to_rank vbench2_beta_long/  →  0 matches
#   grep -rn init_process_group       vbench2_beta_long/  →  0 matches
# Running torchrun --nproc_per_node=N only duplicates work on every rank.
#
# Instead we parallelise at the FOLDER level: each folder is pinned to one
# GPU via CUDA_VISIBLE_DEVICES and we launch up to len(--gpus) in parallel.
# CPU multi-threading won't help either — every dim runs CLIP / DINO /
# DreamSim inference on GPU, so CPU cores are idle during eval.
#
# Usage:
#   bash run_all_folders_eval.sh <output_root> [--gpus 0,1,2,3]
#
# Example:
#   bash run_all_folders_eval.sh /path/to/results --gpus 0,1,2,3

set -e
export TORCH_CUDNN_V8_API_DISABLED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [ $# -lt 1 ]; then
    echo "Usage: bash run_all_folders_eval.sh <output_root> [--gpus 0,1,2,3]"
    exit 1
fi

OUTPUT_ROOT="$1"; shift
GPUS_CSV="0"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS_CSV="$2"; shift 2 ;;
        *) echo "Unknown arg $1"; exit 1 ;;
    esac
done

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
NUM_GPUS=${#GPUS[@]}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTPUT_ROOT"

BASE="/home/zdmaogroup/wubin/reward-forcing-claude-add-experiment-runner-script-PPlnc/reward-forcing-claude-spatial-reward-forcing-L5245/Reward-Forcing-main/videos"

# (label, video_folder) pairs — label is used for the summary grouping.
# Override by exporting FOLDERS_LIST (newline-separated label|path lines)
# before invoking this script.
if [ -n "$FOLDERS_LIST" ]; then
    mapfile -t FOLDERS < <(printf '%s\n' "$FOLDERS_LIST" | sed '/^[[:space:]]*$/d')
else
    FOLDERS=(
        "focalreward_10s|$BASE/focalreward_008000_10s/008000"
        "focalreward_30s|$BASE/focalreward_008000_30s/008000"
        "focalreward_60s|$BASE/focalreward_008000_60s/008000"
        "focalreward_120s|$BASE/focalreward_008000_120s/008000"
        "focalreward_180s|$BASE/focalreward_008000_180s/008000"
        "reward_forcing_10s|$BASE/Reward-Forcing-T2V-1.3B_10s/Reward-Forcing-T2V-1.3B"
        "reward_forcing_30s|$BASE/Reward-Forcing-T2V-1.3B_30s/Reward-Forcing-T2V-1.3B"
        "reward_forcing_60s|$BASE/Reward-Forcing-T2V-1.3B_60s/Reward-Forcing-T2V-1.3B"
        "reward_forcing_120s|$BASE/Reward-Forcing-T2V-1.3B_120s/Reward-Forcing-T2V-1.3B"
        "reward_forcing_180s|$BASE/Reward-Forcing-T2V-1.3B_180s/Reward-Forcing-T2V-1.3B"
    )
fi

DIMS=(subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality)

TOTAL_TASKS=$(( ${#FOLDERS[@]} * ${#DIMS[@]} ))
PROGRESS_FILE="$OUTPUT_ROOT/.progress"
LOCK_FILE="$OUTPUT_ROOT/.stdout.lock"
: > "$PROGRESS_FILE"

export OUTPUT_ROOT PROGRESS_FILE LOCK_FILE TOTAL_TASKS

echo "============================================================"
echo "  VBench Long — Batch Folder Evaluation"
echo "============================================================"
echo "Output root : $OUTPUT_ROOT"
echo "GPUs        : ${GPUS[*]}  (len=$NUM_GPUS)"
echo "Folders     : ${#FOLDERS[@]}"
echo "Dimensions  : ${DIMS[*]}"
echo "Total tasks : $TOTAL_TASKS  (folders × dims)"
echo ""

# ── flock-serialised echo to stdout (workers run concurrently) ──
say() {
    (
        flock 200
        echo "$@"
    ) 200>"$LOCK_FILE"
}
export -f say

# ── Print compact per-folder scores (raw + running total) ────
# Used after every dim finishes so you see scores live.
print_folder_scores() {
    local out="$1"
    local label="$2"
    local just_dim="$3"     # the dim that just finished (underscore form)
    python3 - "$out" "$label" "$just_dim" << 'PYEOF'
import json, os, sys, glob

out, label, just_dim = sys.argv[1], sys.argv[2], sys.argv[3]
just_dim_name = just_dim.replace('_', ' ')

NORMALIZE_DIC = {
    'subject consistency':    {'Min': 0.1462, 'Max': 1.0},
    'background consistency': {'Min': 0.2615, 'Max': 1.0},
    'motion smoothness':      {'Min': 0.7060, 'Max': 0.9975},
    'dynamic degree':         {'Min': 0.0000, 'Max': 1.0},
    'aesthetic quality':      {'Min': 0.0000, 'Max': 1.0},
    'imaging quality':        {'Min': 0.0000, 'Max': 1.0},
}
DIM_WEIGHT = {d: (0.5 if d == 'dynamic degree' else 1.0) for d in NORMALIZE_DIC}
SHORT = {
    'subject consistency':    'SC',
    'background consistency': 'BC',
    'motion smoothness':      'MS',
    'dynamic degree':         'DD',
    'aesthetic quality':      'AQ',
    'imaging quality':        'IQ',
}

raw = {}
iq_detailed = []
for rf in glob.glob(os.path.join(out, 'results_*_eval_results.json')):
    try:
        with open(rf) as f:
            data = json.load(f)
    except Exception:
        continue
    for k, v in data.items():
        dn = k.replace('_', ' ')
        if dn not in NORMALIZE_DIC:
            continue
        if isinstance(v, list) and v:
            raw[dn] = v[0]
            if dn == 'imaging quality' and len(v) > 1:
                iq_detailed = v[1]
        elif isinstance(v, (int, float)):
            raw[dn] = v

# Drift: mean of per-video IQ std across clips
drift_str = ''
if iq_detailed and isinstance(iq_detailed, list):
    from collections import defaultdict
    import numpy as np
    vscores = defaultdict(list)
    for item in iq_detailed:
        if not isinstance(item, dict) or 'video_path' not in item:
            continue
        cp = item['video_path']
        vn = os.path.basename(os.path.dirname(cp)) if 'split_clip' in cp else os.path.basename(cp)
        vscores[vn].append(item['video_results'])
    if vscores:
        stds = [float(np.std(s)) for s in vscores.values() if len(s) > 1]
        if stds:
            drift_str = f' Drift={float(np.mean(stds)):.3f}'

# Total: weighted normalized over dims we already have
num, denom = 0.0, 0.0
for dim in NORMALIZE_DIC:
    if dim not in raw:
        continue
    mn, mx = NORMALIZE_DIC[dim]['Min'], NORMALIZE_DIC[dim]['Max']
    w = DIM_WEIGHT[dim]
    num += (raw[dim] - mn) / (mx - mn) * w
    denom += w
total_str = f'{(num/denom)*100:6.2f}' if denom > 0 else '  --  '
ndims = sum(1 for d in NORMALIZE_DIC if d in raw)

just_val = raw.get(just_dim_name)
just_str = f'{just_val*100:6.2f}' if just_val is not None else '  --  '

cells = []
for dim in NORMALIZE_DIC:
    tag = SHORT[dim]
    if dim in raw:
        cells.append(f'{tag}={raw[dim]*100:5.2f}')
    else:
        cells.append(f'{tag}= -- ')
picture = ' '.join(cells)
print(f'{label}  just={just_dim}={just_str}  |  {picture}  |  Total={total_str} ({ndims}/6){drift_str}')
PYEOF
}
export -f print_folder_scores

# ── Evaluator for one folder on one GPU ──────────────────────
eval_one_folder() {
    local gpu="$1"
    local label="$2"
    local vdir="$3"
    local out="$OUTPUT_ROOT/$label"
    local log="$out/run.log"

    mkdir -p "$out"
    {
        echo "=== [$label] GPU=$gpu dir=$vdir ==="
        echo "start: $(date)"
    } >> "$log" 2>&1

    if [ ! -d "$vdir" ]; then
        echo "MISSING folder: $vdir" >> "$log"
        echo "FAIL_MISSING" > "$out/status"
        say "  !! MISSING folder for $label: $vdir"
        return
    fi

    cd "$SCRIPT_DIR"
    for DIM in "${DIMS[@]}"; do
        # Skip if this dim already produced a results file
        if ls "$out"/results_*_eval_results.json 2>/dev/null \
           | xargs -I{} grep -l "\"${DIM}\"" {} 2>/dev/null | grep -q .; then
            echo "  skip $DIM (already done)" >> "$log"
            # Still bump progress so totals stay correct.
            echo "skip $label $DIM" >> "$PROGRESS_FILE"
            local done_n
            done_n=$(wc -l < "$PROGRESS_FILE")
            local line
            line=$(print_folder_scores "$out" "$label" "$DIM" 2>/dev/null)
            say "[${done_n}/${TOTAL_TASKS}] SKIP  $line"
            continue
        fi
        echo "  -- $DIM" >> "$log"
        local port
        port=$(shuf -i 29500-39999 -n 1)
        # Stream eval output through progress_filter.py so we get live
        # 20/40/60/80% pings in addition to the final DONE line.
        # PYTHONUNBUFFERED=1 keeps tqdm updates flushing in real-time.
        # pipefail so an eval crash actually surfaces here instead of
        # being hidden behind the filter's 0 exit code.
        # Snapshot result files before so we can detect the "exit-0 but
        # wrote nothing" pathological case after the run.
        local before_files
        before_files=$(ls "$out"/results_*_eval_results.json 2>/dev/null | wc -l)
        local eval_rc=0
        (
            set -o pipefail
            CUDA_VISIBLE_DEVICES="$gpu" \
            MASTER_PORT="$port" \
            PYTHONUNBUFFERED=1 \
            python vbench2_beta_long/eval_long.py \
                --videos_path "$vdir" \
                --dimension "$DIM" \
                --mode long_custom_input \
                --output_path "$out" \
                --load_ckpt_from_local True \
                --dev_flag 2>&1 \
              | python3 "$SCRIPT_DIR/progress_filter.py" \
                    --log "$log" \
                    --label "$label" \
                    --dim "$DIM" \
                    --lockfile "$LOCK_FILE"
        ) || eval_rc=$?
        if [ "$eval_rc" -ne 0 ]; then
            echo "  WARNING: $DIM failed on $label (rc=$eval_rc)" >> "$log"
            say "  !! FAIL  $label  $DIM  (rc=$eval_rc) — see $log"
        else
            # Exit 0 but no new results file? Something swallowed the error.
            local after_files
            after_files=$(ls "$out"/results_*_eval_results.json 2>/dev/null | wc -l)
            if [ "$after_files" -le "$before_files" ]; then
                echo "  WARNING: $DIM on $label exited 0 but produced no new results file" >> "$log"
                say "  !! FAIL (no output)  $label  $DIM  — see $log"
            fi
        fi

        # Bump progress + print live scores
        echo "done $label $DIM" >> "$PROGRESS_FILE"
        local done_n
        done_n=$(wc -l < "$PROGRESS_FILE")
        local line
        line=$(print_folder_scores "$out" "$label" "$DIM" 2>/dev/null)
        say "[${done_n}/${TOTAL_TASKS}] DONE  $line"
    done

    {
        echo "done: $(date)"
    } >> "$log" 2>&1
    echo "OK" > "$out/status"
    say "  ==> FOLDER FINISHED: $label"
}
export -f eval_one_folder

# ── Run all folders with GPU-level parallelism ───────────────
pids=()
for i in "${!FOLDERS[@]}"; do
    entry="${FOLDERS[$i]}"
    label="${entry%%|*}"
    vdir="${entry##*|}"
    gpu="${GPUS[$((i % NUM_GPUS))]}"

    # Throttle to NUM_GPUS concurrent jobs
    while (( $(jobs -rp | wc -l) >= NUM_GPUS )); do
        wait -n
    done

    echo "[${i}/${#FOLDERS[@]}] LAUNCH $label on GPU $gpu"
    eval_one_folder "$gpu" "$label" "$vdir" &
    pids+=($!)
done
wait

echo ""
echo "============================================================"
echo "  All folders evaluated — aggregating summary"
echo "============================================================"

# ── Aggregate per folder into one TXT ─────────────────────────
SUMMARY_TXT="$OUTPUT_ROOT/summary.txt"
SUMMARY_JSON="$OUTPUT_ROOT/summary.json"

python3 - "$OUTPUT_ROOT" "$SUMMARY_TXT" "$SUMMARY_JSON" << 'PYEOF'
import json, os, sys, glob, re
from collections import defaultdict

output_root, txt_path, json_path = sys.argv[1], sys.argv[2], sys.argv[3]

# ── VBench normalization (standard VBench coefficients) ──────
NORMALIZE_DIC = {
    'subject consistency':    {'Min': 0.1462, 'Max': 1.0},
    'background consistency': {'Min': 0.2615, 'Max': 1.0},
    'motion smoothness':      {'Min': 0.7060, 'Max': 0.9975},
    'dynamic degree':         {'Min': 0.0000, 'Max': 1.0},
    'aesthetic quality':      {'Min': 0.0000, 'Max': 1.0},
    'imaging quality':        {'Min': 0.0000, 'Max': 1.0},
}
# Following VBench: dynamic_degree is weighted 0.5 so denominator = 5.5
DIM_WEIGHT = {d: (0.5 if d == 'dynamic degree' else 1.0) for d in NORMALIZE_DIC}
ORDER = list(NORMALIZE_DIC.keys())

def load_folder_scores(folder_out):
    """Return ({dim_name: raw_score}, drift_or_None)."""
    scores = {}
    iq_detailed = []
    for rf in glob.glob(os.path.join(folder_out, 'results_*_eval_results.json')):
        with open(rf) as f:
            data = json.load(f)
        for k, v in data.items():
            dn = k.replace('_', ' ')
            if dn not in NORMALIZE_DIC:
                continue
            if isinstance(v, list) and v:
                scores[dn] = v[0]
                if dn == 'imaging quality' and len(v) > 1:
                    iq_detailed = v[1]
            elif isinstance(v, (int, float)):
                scores[dn] = v
    drift = None
    if iq_detailed and isinstance(iq_detailed, list):
        import numpy as np
        vscores = defaultdict(list)
        for item in iq_detailed:
            if not isinstance(item, dict) or 'video_path' not in item:
                continue
            cp = item['video_path']
            vn = os.path.basename(os.path.dirname(cp)) if 'split_clip' in cp else os.path.basename(cp)
            vscores[vn].append(item['video_results'])
        if vscores:
            stds = [float(np.std(s)) for s in vscores.values() if len(s) > 1]
            if stds:
                drift = float(np.mean(stds))
    return scores, drift

def compute_total(raw):
    """Weighted normalized total following VBench Long (denom = 5.5)."""
    if not raw:
        return None, 0
    num = 0.0
    denom = 0.0
    for dim in NORMALIZE_DIC:
        if dim not in raw:
            continue
        mn = NORMALIZE_DIC[dim]['Min']
        mx = NORMALIZE_DIC[dim]['Max']
        w = DIM_WEIGHT[dim]
        num += (raw[dim] - mn) / (mx - mn) * w
        denom += w
    if denom == 0:
        return None, 0
    return num / denom, len([d for d in NORMALIZE_DIC if d in raw])

def parse_label(label):
    """focalreward_180s → ('focalreward', 180). Unknown → (label, 0)."""
    m = re.match(r'(.*)_(\d+)s$', label)
    if m:
        return m.group(1), int(m.group(2))
    return label, 0

# ── Scan every subdir of output_root ─────────────────────────
results = {}
for label in sorted(os.listdir(output_root)):
    sub = os.path.join(output_root, label)
    if not os.path.isdir(sub):
        continue
    raw, drift = load_folder_scores(sub)
    if not raw:
        continue
    total, ndims = compute_total(raw)
    results[label] = {
        'raw': raw,
        'total': total,
        'ndims': ndims,
        'drift': drift,
    }

# ── Group by method, sort by length ──────────────────────────
by_method = defaultdict(list)
for label, r in results.items():
    method, length = parse_label(label)
    by_method[method].append((length, label, r))
for m in by_method:
    by_method[m].sort(key=lambda x: x[0])

# ── Write TXT summary ────────────────────────────────────────
lines = []
lines.append('=' * 100)
lines.append('  VBench Long — Multi-folder Summary')
lines.append('=' * 100)
lines.append('')
lines.append('Notes:')
lines.append('  - Total = sum((raw-Min)/(Max-Min)*w) / sum(w), denom=5.5 (dynamic_degree weight 0.5).')
lines.append('  - Scores printed x100. "MISSING" means that dimension was not produced for the folder.')
lines.append('  - Folders are grouped per method and sorted by video length so different')
lines.append('    lengths can be compared directly within a method.')
lines.append('')

header_dims = [
    ('subject consistency',    'SubjCons'),
    ('background consistency', 'BgCons  '),
    ('motion smoothness',      'MotSmth '),
    ('dynamic degree',         'DynDeg  '),
    ('aesthetic quality',      'Aesthet '),
    ('imaging quality',        'ImgQual '),
]

for method in sorted(by_method):
    lines.append('-' * 100)
    lines.append(f'Method: {method}')
    lines.append('-' * 100)
    hdr = f'  {"Length":>7s}  ' + '  '.join(f'{h:>8s}' for _, h in header_dims) + f'  {"Total":>8s}  {"Drift":>7s}  {"Dims":>5s}'
    lines.append(hdr)
    for length, label, r in by_method[method]:
        raw = r['raw']
        row = f'  {str(length)+"s":>7s}  '
        cells = []
        for dim, _ in header_dims:
            if dim in raw:
                cells.append(f'{raw[dim]*100:8.2f}')
            else:
                cells.append(f'{"--":>8s}')
        row += '  '.join(cells)
        if r['total'] is not None:
            row += f'  {r["total"]*100:8.2f}'
        else:
            row += f'  {"--":>8s}'
        drift = r.get('drift')
        row += f'  {drift:7.3f}' if drift is not None else f'  {"--":>7s}'
        row += f'  {r["ndims"]:>3d}/6' if r['total'] is not None else f'  {"--":>5s}'
        lines.append(row)
    lines.append('')

# Side-by-side comparison across all methods at every length
all_lengths = sorted({l for entries in by_method.values() for (l, _, _) in entries})
if len(by_method) > 1 and all_lengths:
    lines.append('=' * 100)
    lines.append('  Cross-method comparison — Total score')
    lines.append('=' * 100)
    method_names = sorted(by_method.keys())
    hdr = f'  {"Length":>7s}  ' + '  '.join(f'{m:>20s}' for m in method_names)
    lines.append(hdr)
    # Build lookup: method -> {length: total}
    lookup = {m: {l: r['total'] for (l, _, r) in entries} for m, entries in by_method.items()}
    for L in all_lengths:
        row = f'  {str(L)+"s":>7s}  '
        cells = []
        for m in method_names:
            v = lookup[m].get(L)
            cells.append(f'{v*100:20.2f}' if v is not None else f'{"--":>20s}')
        row += '  '.join(cells)
        lines.append(row)
    lines.append('')

txt = '\n'.join(lines) + '\n'
with open(txt_path, 'w') as f:
    f.write(txt)

# Also dump a machine-readable JSON
with open(json_path, 'w') as f:
    json.dump({
        label: {
            'raw_x100': {k: round(v*100, 2) for k, v in r['raw'].items()},
            'total_x100': round(r['total']*100, 2) if r['total'] is not None else None,
            'drift': round(r['drift'], 3) if r.get('drift') is not None else None,
            'ndims': r['ndims'],
        }
        for label, r in results.items()
    }, f, indent=2)

print(txt)
print(f'Wrote: {txt_path}')
print(f'Wrote: {json_path}')
PYEOF

echo ""
echo "==> All done. Summary: $SUMMARY_TXT"
