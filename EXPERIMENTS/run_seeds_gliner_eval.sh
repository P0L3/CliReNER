#!/bin/bash

DATA="P0L3/CliReNER_v_1_1_28_SILVER"
PROJECT="CLIRENER_SILVER_SEEDS"
BASE_NAME="GLiNER_CliSciBERT"

for SEED in 0 42 3012 33 131
do
    MODEL_PATH="EXPERIMENTS/models/GLINER/GLiNER_CliSciBERT_CliReNER_v_1_1_28_SILVER_s${SEED}/checkpoint-final"
    
    echo "-----------------------------------"
    echo "Evaluating Seed $SEED"
    echo "-----------------------------------"
    
    python -m EXPERIMENTS.evaluate \
      --model_type GLINER \
      --dataset_id $DATA \
      --model_path $MODEL_PATH \
      --wandb_project $PROJECT \
      --wandb_run_name "${BASE_NAME}_seed${SEED}"
done