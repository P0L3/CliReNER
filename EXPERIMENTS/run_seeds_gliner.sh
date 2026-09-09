#!/bin/bash

MODEL="EXPERIMENTS/models/GLINER_CUSTOM/clirebert_clivocab_uncased_flips_3es1_6es2/stage1_s301202/GLiNER_CliReBERT_flips_3es1_0es2"
DATA="P0L3/CliReNER_v_1_1_28_SILVER"
CONFIG="EXPERIMENTS/gliner_config.json"
PROJECT="CLIRENER_SILVER_SEEDS"
BASE_NAME="GLiNER_CliReBERT_flips_3es1_0es2"

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