#!/bin/bash

ts=( 0.2 0.3 0.4 0.5 )

for t in "${ts[@]}"; do 
    sbatch --job-name=ig-$t-withldm --output=$WORK/temp/ig-$t-withldm.out run-g.sh --t $t --label_dependent_masking  
done;

ts=( 0.1 0.2 0.3 0.4 0.5 )

for t in "${ts[@]}"; do 
    sbatch --job-name=ig-$t --output=$WORK/temp/ig-$t.out run-g.sh --t $t
done;


