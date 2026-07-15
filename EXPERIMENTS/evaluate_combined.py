import torch
import wandb
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

# --- Import from your existing modules ---
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

SILVER_DATASET_ID = "P0L3/CliReNER_v_1_1_28_SILVER"
GOLD_DATASET_ID = "P0L3/CliReNER_v_1_1_28_GOLD"

# The NEW target WandB Project
WANDB_PROJECT = "CLIRENER_COMBINED_EVAL"

# Seeds to iterate over
SEEDS = [0, 42, 3012, 33, 131]

# List of Models to Evaluate (All 13 Models)
MODELS_TO_EVALUATE = [
    ("SPANMARKER", "FacebookAI/roberta-base"),
    ("SPANMARKER", "nasa-impact/nasa-smd-ibm-v0.1"),
    ("SPANMARKER", "nasa-impact/indus-sde-v0.2"),
    
    ("SPANMARKER", "google-bert/bert-base-uncased"),
    ("SPANMARKER", "allenai/scibert_scivocab_uncased"),
    ("SPANMARKER", "P0L3/cliscibert_scivocab_uncased"),
    ("SPANMARKER", "P0L3/clirebert_clirevocab_uncased"),
    
    ("SPANMARKER", "ESGBERT/EnvironmentalBERT-base"),
    ("SPANMARKER", "distilbert/distilroberta-base"),
    ("SPANMARKER", "climatebert/distilroberta-base-climate-f"),
    ("SPANMARKER", "P0L3/sciclimatebert"),
    
    ("GLINER", "gliner-community/gliner_medium-v2.5"),
    ("GLINER", "gliner-community/gliner_small-v2.5")
]


def load_and_combine_data(gold_id, silver_id):
    """
    Loads all splits of the Gold dataset and the test split of the Silver dataset,
    then combines them into a single dataset mapped to 'test' for evaluation.
    """
    print(f"--- Loading GOLD Dataset: {gold_id} ---")
    ds_gold = load_dataset(gold_id)
    splits_to_merge_gold = [ds_gold[split] for split in ds_gold.keys()]
    merged_gold = concatenate_datasets(splits_to_merge_gold)
    print(f"Merged Gold splits {list(ds_gold.keys())} | Size: {len(merged_gold)}")

    print(f"--- Loading SILVER Dataset (Test Split Only): {silver_id} ---")
    ds_silver = load_dataset(silver_id)
    silver_test = ds_silver["test"]
    print(f"Loaded Silver Test split | Size: {len(silver_test)}")
    
    # Combine Gold (All) + Silver (Test)
    combined_dataset = concatenate_datasets([merged_gold, silver_test])
    print(f"--- COMBINED EVALUATION DATASET CREATED | Total Size: {len(combined_dataset)} ---")
    
    # Get the feature names for mapping
    bio_label_list = combined_dataset.features["ner_tags"].feature.names

    return {"test": combined_dataset}, bio_label_list


def run_campaign():
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # 2. Prepare Combined Data (Load once to save time)
    dataset_dict, bio_label_list = load_and_combine_data(GOLD_DATASET_ID, SILVER_DATASET_ID)
    
    target_labels = list(CLIRENER_LABELS_V1)

    # 3. Iterate Models and Seeds
    for model_type, model_id in MODELS_TO_EVALUATE:
        for seed in SEEDS:
            
            # A. Find the local path to the trained Silver model
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
            run_name = f"eval_COMBINED_{shorten_name(model_id)}_s{seed}"
            
            run = wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                reinit=True,
                config={
                    "model_type": model_type,
                    "model_id": model_id,
                    "training_dataset": SILVER_DATASET_ID,
                    "evaluation_dataset": f"COMBINED_GOLD_ALL_AND_SILVER_TEST", 
                    "seed": seed,
                    "evaluation_scope": "GOLD(ALL) + SILVER(TEST)"
                }
            )

            try:
                # C. Run Inference
                raw_predictions = []
                
                if model_type == "GLINER":
                    raw_predictions = evaluate_gliner(str(checkpoint_path), dataset_dict, target_labels, device)
                elif model_type == "SPANMARKER":
                    raw_predictions = evaluate_spanmarker(str(checkpoint_path), dataset_dict, target_labels, device)

                # D. Transform Predictions
                print("--- Transforming Predictions ---")
                model_predictions_transformed = transform_to_ner_format(raw_predictions, target_labels)
                
                pred_ids = [row["ner_tags"] for row in model_predictions_transformed[0]]
                
                # Get True IDs from the combined dataset
                true_ids = dataset_dict["test"]["ner_tags"]

                # E. Calculate Metrics
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