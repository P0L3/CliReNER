import argparse
import torch
import wandb
import sys
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, DatasetDict

# --- Import from your existing modules ---
# Ensure these files are accessible in the python path
from EXPERIMENTS.finetune import get_output_dir, shorten_name
from EXPERIMENTS.evaluate import (
    evaluate_gliner, 
    evaluate_spanmarker, 
    transform_to_ner_format, 
    run_nervaluate, 
    log_to_wandb,
    CLIRENER_LABELS_V1
)

# --- CONFIGURATION ---

# 1. The Dataset used for TRAINING (used to find where models are saved)
SILVER_DATASET_ID = "P0L3/CliReNER_v_1_1_28_SILVER"

# 2. The Dataset used for EVALUATION (The new target)
GOLD_DATASET_ID = "P0L3/CliReNER_v_1_1_28_GOLD_authorannots"

# 3. Target WandB Project
WANDB_PROJECT = "CLIRENER_GOLD_SEEDS_authorannots"

# 4. Seeds to iterate over
SEEDS = [0, 42, 3012, 33, 131]

# 5. List of Models to Evaluate. 
# FORMAT: ("MODEL_TYPE", "HuggingFace_Model_ID")
# Add all 11 models here.
MODELS_TO_EVALUATE = [
    ("SPANMARKER", "FacebookAI/roberta-base"),
    
    ("SPANMARKER", "google-bert/bert-base-uncased"),
    ("SPANMARKER", "ESGBERT/EnvironmentalBERT-base"),
    ("SPANMARKER", "allenai/scibert_scivocab_uncased"),
    ("SPANMARKER", "P0L3/cliscibert_scivocab_uncased"),
    ("SPANMARKER", "P0L3/clirebert_clirevocab_uncased"),
    
    ("SPANMARKER", "distilbert/distilroberta-base"),
    ("SPANMARKER", "climatebert/distilroberta-base-climate-f"),
    ("SPANMARKER", "P0L3/sciclimatebert"),
    
    ("GLINER", "gliner-community/gliner_medium-v2.5"),
    ("GLINER", "gliner-community/gliner_small-v2.5")
]

def load_and_merge_gold_data(dataset_id):
    """
    Loads the dataset and merges Train + Validation + Test into a single split
    assigned to the key 'test' so existing evaluate functions accept it.
    """
    print(f"--- Loading and Merging GOLD Dataset: {dataset_id} ---")
    ds = load_dataset(dataset_id)
    
    # List all available splits (usually train, validation, test)
    splits_to_merge = [ds[split] for split in ds.keys()]
    
    merged_dataset = concatenate_datasets(splits_to_merge)
    print(f"Merged {list(ds.keys())} into a single dataset of size: {len(merged_dataset)}")
    
    # Return as a dict with "test" key to satisfy evaluate.py requirements
    return {"test": merged_dataset}, merged_dataset.features["ner_tags"].feature.names

def run_campaign():
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    # 2. Prepare Data (Load once to save time)
    dataset_dict, bio_label_list = load_and_merge_gold_data(GOLD_DATASET_ID)
    
    target_labels = list(CLIRENER_LABELS_V1)

    # 3. Iterate Models and Seeds
    for model_type, model_id in MODELS_TO_EVALUATE:
        for seed in SEEDS:
            
            # A. Reconstruct the path where the TRAINED (Silver) model is saved
            # Note: We look for the folder based on SILVER_DATASET_ID
            base_output_dir = Path("EXPERIMENTS/models")
            saved_model_dir = get_output_dir(base_output_dir, model_type, model_id, SILVER_DATASET_ID, seed)
            checkpoint_path = saved_model_dir / "checkpoint-final"
            
            print(f"\n{'#'*60}")
            print(f"Processing: {model_id} | Seed: {seed}")
            print(f"Looking for checkpoint: {checkpoint_path}")
            
            if not checkpoint_path.exists():
                print(f"!!! SKIPPING: Model checkpoint not found at {checkpoint_path}")
                continue

            # B. Initialize WandB Run
            run_name = f"eval_GOLD_{shorten_name(model_id)}_s{seed}"
            
            # Start a fresh run for this evaluation
            run = wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                reinit=True, # Allow multiple runs in one script
                config={
                    "model_type": model_type,
                    "model_id": model_id,
                    "training_dataset": SILVER_DATASET_ID,
                    "evaluation_dataset": GOLD_DATASET_ID, # Explicitly logging this
                    "seed": seed,
                    "evaluation_scope": "ALL_SPLITS_MERGED"
                }
            )

            try:
                # C. Run Inference (Reusing logic from evaluate.py)
                raw_predictions = []
                
                if model_type == "GLINER":
                    raw_predictions = evaluate_gliner(str(checkpoint_path), dataset_dict, target_labels, device)
                elif model_type == "SPANMARKER":
                    raw_predictions = evaluate_spanmarker(str(checkpoint_path), dataset_dict, target_labels, device)

                # D. Transform Predictions
                print("--- Transforming Predictions ---")
                model_predictions_transformed = transform_to_ner_format(raw_predictions, target_labels)
                
                pred_ids = [row["ner_tags"] for row in model_predictions_transformed[0]]
                
                # Get True IDs from the merged dataset
                true_ids = dataset_dict["test"]["ner_tags"]

                # E. Calculate Metrics
                # We pass the dataset's BIO scheme for ID mapping, and the target labels for reporting
                results, results_by_tag = run_nervaluate(true_ids, pred_ids, bio_label_list, tags=target_labels)

                # F. Log to WandB
                log_to_wandb(results, results_by_tag)
                
                print(f"SUCCESS: {run_name}")

            except Exception as e:
                print(f"!!! ERROR evaluating {run_name}: {e}")
                wandb.log({"error": str(e)})
            
            finally:
                wandb.finish()

if __name__ == "__main__":
    run_campaign()