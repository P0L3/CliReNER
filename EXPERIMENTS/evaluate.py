import argparse
import torch
import os
import pandas as pd
import wandb
from datasets import load_dataset
from nervaluate import Evaluator
from dataset_processing import transform_to_ner_format

# Import model specifics
from gliner import GLiNER
from span_marker import SpanMarkerModel

def ids_to_labels(pred_id_seqs, label_list):
    """
    Convert sequences of prediction IDs into label sequences.
    """
    return [[label_list[i] for i in seq] for seq in pred_id_seqs]

def evaluate_gliner(model_path, dataset, labels, device):
    """
    Run inference using GLiNER model.
    """
    print(f"--- Loading GLiNER model from {model_path} ---")
    model = GLiNER.from_pretrained(model_path)
    model.to(device)
    
    print("--- Running Inference ---")
    model_predictions = []
    
    # Using the loop strategy from your snippet
    # Note: GLiNER can accept lists for batching, but we keep your logic for safety
    for row in dataset["test"]:
        text = row["text"]
        # We predict using the labels found in the dataset features
        entities = model.predict_entities(text, labels, threshold=0.5) # threshold adjusted typically, or use 0.1 as per snippet
        model_predictions.append({
            "text": text,
            "entities": entities
        })
        
    return model_predictions

def evaluate_spanmarker(model_path, dataset, labels, device):
    """
    Run inference using SpanMarker model.
    """
    print(f"--- Loading SpanMarker model from {model_path} ---")
    # SpanMarker handles device automatically usually, but good to be explicit if needed
    model = SpanMarkerModel.from_pretrained(model_path)
    if torch.cuda.is_available():
        model.cuda()
    
    print("--- Running Inference ---")
    # Extract text list
    text_list = [row["text"] for row in dataset["test"]]
    
    # Batch prediction
    entities_list = model.predict(text_list)
    
    # Formatting to match the structure expected by transform_to_ner_format
    model_predictions = []
    for i, row_entities in enumerate(entities_list):
        row_text = text_list[i]
        formatted_entities = []
        for entity in row_entities:
            formatted_entities.append({
                'start': entity["char_start_index"],
                'end': entity["char_end_index"],
                'text': entity["span"],
                'label': entity["label"],
                'score': entity["score"]
            })
            
        model_predictions.append({
            "text": row_text,
            "entities": formatted_entities
        })
        
    return model_predictions

def run_nervaluate(true_ids, pred_ids, label_list, tags):
    """
    Compare Ground Truth vs Predictions using Nervaluate.
    """
    print("--- Calculating Metrics ---")
    true_labels = ids_to_labels(true_ids, label_list)
    pred_labels = ids_to_labels(pred_ids, label_list)
    
    evaluator = Evaluator(true_labels, pred_labels, tags=tags, loader="list")
    results, results_by_tag, _, _ = evaluator.evaluate()
    
    return results, results_by_tag

def log_to_wandb(results, results_by_tag):
    """
    Log metrics and dataframes to WandB.
    """
    # 1. Log overall metrics (Strict & Exact)
    wandb.log({
        "strict_precision": results["strict"]["precision"],
        "strict_recall": results["strict"]["recall"],
        "strict_f1": results["strict"]["f1"],
        "exact_precision": results["exact"]["precision"],
        "exact_recall": results["exact"]["recall"],
        "exact_f1": results["exact"]["f1"],
    })
    
    # 2. Log full results table (Strict, Exact, Partial, Type)
    df_results = pd.DataFrame(results)
    # Transpose for better readability in wandb table
    wandb.log({"overall_results_table": wandb.Table(dataframe=df_results)})
    
    # 3. Log results by tag
    # results_by_tag is a list of dicts or dict of dicts depending on version, 
    # typically nervaluate returns a dict where keys are tags.
    # We convert it to a flat dataframe for WandB.
    
    tag_data = []
    for tag, metrics in results_by_tag.items():
        row = {"tag": tag}
        # Flatten the nested dict (e.g., strict: {p, r, f1})
        for eval_type in ["strict", "exact"]: 
            if eval_type in metrics:
                row[f"{eval_type}_f1"] = metrics[eval_type]["f1"]
                row[f"{eval_type}_p"] = metrics[eval_type]["precision"]
                row[f"{eval_type}_r"] = metrics[eval_type]["recall"]
        tag_data.append(row)
        
    if tag_data:
        df_tags = pd.DataFrame(tag_data)
        wandb.log({"per_tag_results": wandb.Table(dataframe=df_tags)})
    
    print("\n--- Results Logged to WandB ---")
    print(df_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate NER models (GLiNER or SpanMarker) and log to WandB.')
    
    parser.add_argument("--model_type", type=str, required=True, choices=["GLINER", "SPANMARKER"])
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved checkpoint")
    
    # WandB specific arguments
    parser.add_argument("--wandb_project", type=str, default="ner-evaluation", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB username/org")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional name for the run")

    args = parser.parse_args()

    # 1. Device Setup
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda:0')
    else:
        print("CUDA not available. Using CPU.")
        device = torch.device('cpu')

    # 2. Initialize WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name if args.wandb_run_name else f"eval-{args.model_type}-{args.dataset_id.split('/')[-1]}",
        config=vars(args)
    )

    # 3. Load Data
    print(f"Loading dataset: {args.dataset_id}")
    dataset = load_dataset(args.dataset_id)
    
    # Extract labels (ground truth schema)
    # Assumes "ner_tags" features exist as per your training script
    label_list = dataset["train"].features["ner_tags"].feature.names
    
    # Ground Truth IDs
    true_ids = dataset["test"]["ner_tags"]

    # 4. Run Inference
    if args.model_type == "GLINER":
        # Note: We pass label_list here. If you need specific subset like CLIRENER_LABELS_V1,
        # you can filter label_list here.
        raw_predictions = evaluate_gliner(args.model_path, dataset, label_list, device)
    elif args.model_type == "SPANMARKER":
        raw_predictions = evaluate_spanmarker(args.model_path, dataset, label_list, device)

    # 5. Transform Predictions to Aligned format
    # This transforms the "entities" dicts back into aligned BIO/ID sequences
    print("--- Transforming Predictions to format ---")
    model_predictions_transformed = transform_to_ner_format(raw_predictions, label_list)

    pred_ids = []
    for row in model_predictions_transformed[0]:
        pred_ids.append(row["ner_tags"])

    # 6. Evaluate
    # Use the labels found in dataset as the tags for nervaluate
    results, results_by_tag = run_nervaluate(true_ids, pred_ids, label_list, tags=label_list)

    # 7. Log
    log_to_wandb(results, results_by_tag)
    
    wandb.finish()