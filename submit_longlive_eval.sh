#!/bin/bash
#SBATCH -J vb_longlive
#SBATCH -o job-longlive-%j.log
#SBATCH -e job-longlive-%j.err
#SBATCH -p GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#
# LongLive video evaluation. Submits one job that:
#   1. Iterates the 10 LongLive folders (gen_{10,30,60,120}s_{base,lora}
#      + long_180s{,_lora})
#   2. Runs the 6 quality dims of VBench Long on each folder, with up to
#      4 folders in parallel (one per GPU)
#   3. Writes per-folder json + a global summary.txt / summary.json
#
# Notes
# -----
# * Videos are NOT pre-split here — vbench2_beta_long.preprocess will
#   auto-split into 2s clips on first dim, using up to 16 CPU workers
#   from the GPU node. If the 120s/180s folders are slow to split,
#   adapt submit_split_videos.sh for these paths and run that on a CPU
#   node first.
# * Already-completed dims are skipped on re-runs (per-dim granularity),
#   so just re-sbatch this script if anything fails.

echo "Time is $(date)"
echo "Directory is $PWD"
echo "This job runs on the following nodes: $SLURM_JOB_NODELIST"
echo ""

cd /home/zdmaogroup/wubin/reward-forcing-claude-add-experiment-runner-script-PPlnc/vbench
source /home/zdmaogroup/wubin/miniconda3/etc/profile.d/conda.sh
conda activate rl
ulimit -u 65536

export TORCH_CUDNN_V8_API_DISABLED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LL_BASE="/home/zdmaogroup/wubin/reward-forcing-claude-add-experiment-runner-script-PPlnc/reward-forcing-claude-spatial-reward-forcing-L5245/LongLive-main/videos"
OUTPUT_ROOT="/home/zdmaogroup/wubin/reward-forcing-claude-add-experiment-runner-script-PPlnc/vbench_long_results_longlive"

# Note: 180s folders use a different prefix ("long_") than the others ("gen_").
export FOLDERS_LIST="$(cat <<EOF
longlive_base_10s|$LL_BASE/gen_10s_base
longlive_base_30s|$LL_BASE/gen_30s_base
longlive_base_60s|$LL_BASE/gen_60s_base
longlive_base_120s|$LL_BASE/gen_120s_base
longlive_base_180s|$LL_BASE/long_180s
longlive_lora_10s|$LL_BASE/gen_10s_lora
longlive_lora_30s|$LL_BASE/gen_30s_lora
longlive_lora_60s|$LL_BASE/gen_60s_lora
longlive_lora_120s|$LL_BASE/gen_120s_lora
longlive_lora_180s|$LL_BASE/long_180s_lora
EOF
)"

bash run_all_folders_eval.sh "$OUTPUT_ROOT" --gpus 0,1,2,3

echo ""
echo "Finished at $(date)"
