#!/bin/bash

MODEL="gliner-community/gliner_medium-v2.5"
DATA="P0L3/CliReNER_v_1_1_28_SILVER"
CONFIG="EXPERIMENTS/gliner_config.json"
PROJECT="CLIRENER_SILVER_SEEDS"
BASE_NAME="GLiNER_Medium"

# Loop through seeds
for SEED in 0 42 3012 33 131
do
    echo "-----------------------------------"
    echo "Running Seed $SEED"
    echo "-----------------------------------"
    
    python -m EXPERIMENTS.finetune_evaluate_pipeline \
      --model_type GLINER \
      --dataset_id $DATA \
      --model_id $MODEL \
      --config_path $CONFIG \
      --wandb_project $PROJECT \
      --wandb_name "${BASE_NAME}" \
      --seed $SEED
done