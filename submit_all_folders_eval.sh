#!/bin/bash
#SBATCH -J vb_long_all
#SBATCH -o job-vblong-%j.log
#SBATCH -e job-vblong-%j.err
#SBATCH -p GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

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

OUTPUT_ROOT="/home/zdmaogroup/wubin/reward-forcing-claude-add-experiment-runner-script-PPlnc/vbench_long_results_all"

# Use all 4 GPUs at the folder level (long eval has no intra-dim multi-GPU).
bash run_all_folders_eval.sh "$OUTPUT_ROOT" --gpus 0,1,2,3

echo ""
echo "Finished at $(date)"
