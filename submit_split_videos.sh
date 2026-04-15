#!/bin/bash
#SBATCH -J split_vids
#SBATCH -o job-split-%j.log
#SBATCH -e job-split-%j.err
#SBATCH -p CPU-192C768GB
#SBATCH -c 128
#SBATCH --mem=256G
#SBATCH --time=8:00:00

echo "Time is $(date)"
echo "Directory is $PWD"
echo "This job runs on the following nodes: $SLURM_JOB_NODELIST"
echo ""

cd /home/zdmaogroup/wubin/reward-forcing-claude-add-experiment-runner-script-PPlnc/vbench
source /home/zdmaogroup/wubin/miniconda3/etc/profile.d/conda.sh
conda activate rl
ulimit -u 65536

# Limit BLAS/OMP threads so worker processes don't fight each other
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BASE="/home/zdmaogroup/wubin/reward-forcing-claude-add-experiment-runner-script-PPlnc/reward-forcing-claude-spatial-reward-forcing-L5245/Reward-Forcing-main/videos"

FOLDERS=(
    "$BASE/focalreward_008000_10s"
    "$BASE/focalreward_008000_30s"
    "$BASE/focalreward_008000_60s"
    "$BASE/focalreward_008000_120s"
    "$BASE/focalreward_008000_180s"
    "$BASE/Reward-Forcing-T2V-1.3B_10s"
    "$BASE/Reward-Forcing-T2V-1.3B_30s"
    "$BASE/Reward-Forcing-T2V-1.3B_60s"
    "$BASE/Reward-Forcing-T2V-1.3B_120s"
    "$BASE/Reward-Forcing-T2V-1.3B_180s"
)

python3 batch_split_videos.py --workers 96 --duration 2 "${FOLDERS[@]}"

echo ""
echo "Finished at $(date)"
