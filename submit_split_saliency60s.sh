#!/bin/bash
#SBATCH -J split_sal60
#SBATCH -o job-split-sal60-%j.log
#SBATCH -e job-split-sal60-%j.err
#SBATCH -p CPU-192C768GB
#SBATCH --qos=qos_cpu_192c768gb
#SBATCH -c 64
#SBATCH --mem=700G
#SBATCH --time=4:00:00

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

FOLDERS=(
    "/home/zdmaogroup/wubin/reward-forcing-claude-add-experiment-runner-script-PPlnc/reward-forcing-claude-spatial-reward-forcing-L5245/evaluate/20260429_051102_saliency_only_60s/008000"
)

python3 batch_split_videos.py --workers 64 --duration 2 --mem-budget-gb 600 "${FOLDERS[@]}"

echo ""
echo "Finished at $(date)"
