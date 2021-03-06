#!/bin/sh
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --constraint='gpu_32gb&gpu_v100'
#SBATCH --job-name=baseline
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=atendle13.3.98@gmail.com
#SBATCH --output=/work/cse896/atendle/out-files/random70-test.out

export HF_DATASETS_CACHE=/work/cse896/atendle/.cache/datasets/
export TRANSFORMERS_CACHE=/work/cse896/atendle/.cache/transformers/
export HF_MODULES_CACHE=/work/cse896/atendle/.cache/huggingface/
module load anaconda
conda activate /work/vinod/gwirka/.conda/envs/nlp-pbi

#python -u $@ tfidf_baseline.py --epochs 10 --dataset IMDb
# python -u $@ random_baseline.py --epochs 10 --dataset IMDb
#python -u $@ baseline.py --epochs 20 --dataset IMDb
python -u $@ test.py --model-type "random" --model "/work/cse896/atendle/model-files/random_percent/random_70" --tokenizer "/work/cse896/atendle/model-files/random_percent/random_70-tok"
