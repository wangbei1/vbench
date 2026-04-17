#!/bin/bash
#SBATCH -J split_ll
#SBATCH -o job-split-ll-%j.log
#SBATCH -e job-split-ll-%j.err
#SBATCH -p CPU-96C3TB
#SBATCH --qos=qos_cpu_96c3tb
#SBATCH -c 64
#SBATCH --mem=2500G
#SBATCH --time=8:00:00

echo "Time is $(date)"
echo "Directory is $PWD"
echo "This job runs on the following nodes: $SLURM_JOB_NODELIST"
echo ""

cd /home/zdmaogroup/wubin/reward-forcing-claude-add-experiment-runner-script-PPlnc/vbench
source /home/zdmaogroup/wubin/miniconda3/etc/profile.d/conda.sh
conda activate rl
ulimit -u 65536

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BASE="/home/zdmaogroup/wubin/reward-forcing-claude-add-experiment-runner-script-PPlnc/reward-forcing-claude-spatial-reward-forcing-L5245/LongLive-main/videos"

FOLDERS=(
    "$BASE/gen_10s_base"
    "$BASE/gen_10s_lora"
    "$BASE/gen_30s_base"
    "$BASE/gen_30s_lora"
    "$BASE/gen_60s_base"
    "$BASE/gen_60s_lora"
    "$BASE/gen_120s_base"
    "$BASE/gen_120s_lora"
    "$BASE/long_180s"
    "$BASE/long_180s_lora"
)

python3 batch_split_videos.py --workers 64 --duration 2 --mem-budget-gb 2200 "${FOLDERS[@]}"

echo ""
echo "Finished at $(date)"
