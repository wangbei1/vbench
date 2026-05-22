#!/bin/bash
#SBATCH -J vb_full
#SBATCH -o job-vbfull-%j.log
#SBATCH -e job-vbfull-%j.err
#SBATCH -p GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#
# Full VBench (short) evaluation on 8 GPUs.
# Runs all 16 dimensions and computes Quality / Semantic / Total.
#
# Videos MUST be generated from the official VBench prompt suite
# (prompts/all_dimension.txt) and named XXXX_seedY.mp4, where XXXX
# is the 0-padded prompt index into all_dimension.txt.

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

VIDEO_DIR="/home/zdmaogroup/wubin/reward-forcing-claude-add-experiment-runner-script-PPlnc/reward-forcing-claude-spatial-reward-forcing-L5245/evaluate/20260429_051102_saliency_only/008000"

bash run_full_eval.sh "$VIDEO_DIR" --ngpus 8

echo ""
echo "Finished at $(date)"
