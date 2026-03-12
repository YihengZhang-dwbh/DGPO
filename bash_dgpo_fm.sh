#!/bin/bash

source activate DGPO

id=0
env_name="CheetahRun"
seed=1
time=$(date '+%Y-%m-%d-%H%M%S')

mkdir -p logs

echo "🚀 [$(date '+%H:%M:%S')] is ready to run DGPO in ${env_name} on GPU ${id}"

echo "Run DGPO (Flow Matching)... > logs/train_dgpo_fm_${env_name}_${time}.txt"
CUDA_VISIBLE_DEVICES=$id python scripts/train_dgpo_fm.py \
    --env-name $env_name \
    --seed $seed \
    > logs/train_dgpo_fm_${env_name}_${time}.txt 2>&1
