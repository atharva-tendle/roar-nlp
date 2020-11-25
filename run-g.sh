#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH --partition=gpu,scott,cse896
#SBATCH --gres=gpu
#SBATCH --constraint='gpu_32gb&gpu_v100'
#SBATCH --mem-per-cpu=0
#SBATCH --mem=150G
#SBATCH --job-name=ig-train
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=gwirka@gmail.com
#SBATCH --output=/work/vinod/gwirka/temp/%J.out
#SBATCH --error=/work/vinod/gwirka/temp/%J.err

export HF_DATASETS_CACHE=/work/vinod/gwirka/.cache/datasets/
export HF_MODULES_CACHE=/work/vinod/gwirka/.cache/huggingface/
export TRANSFORMERS_CACHE=/work/vinod/gwirka/.cache/transformers/
export MPLCONFIGDIR=/work/vinod/gwirka/.cache/mlp/

module load anaconda
conda activate /work/vinod/gwirka/.conda/envs/nlp-pbi

python -u ig_baseline.py "$@"
