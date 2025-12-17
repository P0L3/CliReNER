import torch
import wandb
from datasets import load_dataset, concatenate_datasets

# --- Import from your existing modules ---
from EXPERIMENTS.finetune import shorten_name
from EXPERIMENTS.evaluate import (
    evaluate_gliner, 
    transform_to_ner_format, 
    run_nervaluate, 
    log_to_wandb,
    CLIRENER_LABELS_V1
)

# --- CONFIGURATION ---

# 1. Dataset
GOLD_DATASET_ID = "P0L3/CliReNER_v_1_1_28_GOLD_authorannots"

# 2. WandB Project (Suggested to keep separate from fine-tuned runs)
WANDB_PROJECT = "CLIRENER_GOLD_SEEDS_authorannots"

# 3. Base GLiNER models to fetch from Hugging Face Hub
MODELS_TO_EVALUATE = [
    "gliner-community/gliner_medium-v2.5",
    "gliner-community/gliner_small-v2.5",
    # Add others if needed, e.g., "urchade/gliner_large-v2.1"
]

def load_and_merge_gold_data(dataset_id):
    """
    Loads the dataset and merges Train + Validation + Test into a single split.
    """
    print(f"--- Loading and Merging GOLD Dataset: {dataset_id} ---")
    ds = load_dataset(dataset_id)
    splits_to_merge = [ds[split] for split in ds.keys()]
    merged_dataset = concatenate_datasets(splits_to_merge)
    print(f"Merged {list(ds.keys())} into a single dataset of size: {len(merged_dataset)}")
    
    # Return as dict with "test" key for compatibility
    return {"test": merged_dataset}, merged_dataset.features["ner_tags"].feature.names

def run_zeroshot_eval():
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # 2. Prepare Data
    dataset_dict, bio_label_list = load_and_merge_gold_data(GOLD_DATASET_ID)
    target_labels = list(CLIRENER_LABELS_V1)

    # 3. Iterate Models (No Seeds, Direct from Hub)
    for model_id in MODELS_TO_EVALUATE:
        
        print(f"\n{'#'*60}")
        print(f"Processing Zero-Shot: {model_id}")
        
        run_name = f"eval_GOLD_ZS_{shorten_name(model_id)}"
        
        # Initialize WandB
        run = wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            reinit=True,
            config={
                "model_type": "GLINER",
                "model_id": model_id,
                "training_type": "ZERO_SHOT", 
                "evaluation_dataset": GOLD_DATASET_ID,
                "evaluation_scope": "ALL_SPLITS_MERGED"
            }
        )

        try:
            # A. Run Inference
            # We pass the HF Hub ID directly. GLiNER will download it automatically.
            raw_predictions = evaluate_gliner(model_id, dataset_dict, target_labels, device)

            # B. Transform Predictions
            print("--- Transforming Predictions ---")
            model_predictions_transformed = transform_to_ner_format(raw_predictions, target_labels)
            
            pred_ids = [row["ner_tags"] for row in model_predictions_transformed[0]]
            true_ids = dataset_dict["test"]["ner_tags"]

            # C. Calculate Metrics
            results, results_by_tag = run_nervaluate(true_ids, pred_ids, bio_label_list, tags=target_labels)

            # D. Log to WandB
            log_to_wandb(results, results_by_tag)
            
            print(f"SUCCESS: {run_name}")

        except Exception as e:
            print(f"!!! ERROR evaluating {run_name}: {e}")
            wandb.log({"error": str(e)})
        
        finally:
            wandb.finish()

if __name__ == "__main__":
    run_zeroshot_eval()