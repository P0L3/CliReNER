import argparse
import torch
import os
import pandas as pd
import wandb
from datasets import load_dataset
from nervaluate import Evaluator
from dataset_processing import transform_to_ner_format, CLIRENER_LABELS_V1

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
    import pandas as pd
    
    # 1. Log overall metrics (Strict & Exact)
    wandb.log({
        "overall/strict_precision": results["strict"]["precision"],
        "overall/strict_recall": results["strict"]["recall"],
        "overall/strict_f1": results["strict"]["f1"],
        "overall/exact_precision": results["exact"]["precision"],
        "overall/exact_recall": results["exact"]["recall"],
        "overall/exact_f1": results["exact"]["f1"],
        "overall/partial_precision": results["partial"]["precision"],
        "overall/partial_recall": results["partial"]["recall"],
        "overall/partial_f1": results["partial"]["f1"],
        "overall/type_precision": results["ent_type"]["precision"],
        "overall/type_recall": results["ent_type"]["recall"],
        "overall/type_f1": results["ent_type"]["f1"],
    })
    
    # 2. Log full results table (Strict, Exact, Partial, Type)
    df_results = pd.DataFrame(results)
    wandb.log({"tables/overall_results": wandb.Table(dataframe=df_results)})
    
    # 3. Log Per-Tag Table (Useful for textual inspection)
    tag_data = []
    flattened_metrics = {}

    for tag, metrics in results_by_tag.items():
        # A. Prepare data for the Table
        row = {"tag": tag}
        for eval_type in ["strict", "exact", "partial", "ent_type"]: 
            if eval_type in metrics:
                row[f"{eval_type}_f1"] = metrics[eval_type]["f1"]
                row[f"{eval_type}_p"] = metrics[eval_type]["precision"]
                row[f"{eval_type}_r"] = metrics[eval_type]["recall"]
                row[f"{eval_type}_count_correct"] = metrics[eval_type].get("correct", 0)
                row[f"{eval_type}_count_missed"] = metrics[eval_type].get("missed", 0)
        tag_data.append(row)

        # B. Prepare Flattened Scalar Metrics for Dashboard Visualization
        # We focus on 'strict' for charts to keep it clean, but you can add 'exact' if needed.
        if 'strict' in metrics:
            clean_tag = tag.replace(" ", "_") # Sanitize tag for wandb key
            
            # Key Metrics
            flattened_metrics[f"tag_f1/{clean_tag}"] = metrics['strict']['f1']
            flattened_metrics[f"tag_precision/{clean_tag}"] = metrics['strict']['precision']
            flattened_metrics[f"tag_recall/{clean_tag}"] = metrics['strict']['recall']
            
            # Error Counts (Great for Stacked Bar Charts in Dashboard)
            flattened_metrics[f"tag_correct/{clean_tag}"] = metrics['strict'].get("correct", 0)
            flattened_metrics[f"tag_incorrect/{clean_tag}"] = metrics['strict'].get("incorrect", 0)
            flattened_metrics[f"tag_missed/{clean_tag}"] = metrics['strict'].get("missed", 0)
            flattened_metrics[f"tag_spurious/{clean_tag}"] = metrics['strict'].get("spurious", 0)
        
    if tag_data:
        df_tags = pd.DataFrame(tag_data)
        wandb.log({"tables/per_tag_results": wandb.Table(dataframe=df_tags)})

    # 4. Log the Flattened Metrics
    # This enables "Group By" and "Compare" features in the WandB UI
    wandb.log(flattened_metrics)
    
    print("\n--- Results Logged to WandB ---")
    print(df_results)

def log_ner_visualizations(results_by_tag):
    """
    Generates rich visualizations for NER evaluation.
    1. Per-Tag F1 Score (Bar Chart)
    2. Error Breakdown Counts (Grouped Bar Chart)
    """
    import pandas as pd
    
    # --- Prepare Data ---
    f1_data = []
    error_data = []
    
    # Iterate through each tag (e.g., 'Person', 'Location')
    for tag, metrics in results_by_tag.items():
        # 1. Extract F1 data (using 'strict' mode usually is best for comparison)
        if 'strict' in metrics:
            f1_data.append([tag, metrics['strict']['f1']])
            
        # 2. Extract Error Counts (using 'strict' counts)
        # We focus on: correct, incorrect, missed, spurious
        if 'strict' in metrics:
            m = metrics['strict']
            error_data.append([tag, "Correct", m.get("correct", 0)])
            error_data.append([tag, "Incorrect", m.get("incorrect", 0)])
            error_data.append([tag, "Missed", m.get("missed", 0)])
            error_data.append([tag, "Spurious", m.get("spurious", 0)])

    # --- Visualization 1: F1 Score per Tag ---
    # Convert to Table
    table_f1 = wandb.Table(data=f1_data, columns=["Entity", "Strict F1"])
    
    # Create Bar Plot
    # We sort by F1 score for readability
    f1_data.sort(key=lambda x: x[1], reverse=True)
    bar_plot_f1 = wandb.plot.bar(
        table_f1, "Entity", "Strict F1", 
        title="Strict F1 Score by Entity Type"
    )
    
    wandb.log({"chart_f1_per_tag": bar_plot_f1})

    # --- Visualization 2: Error Analysis (Grouped Bar Chart) ---
    # This helps diagnose: Are we missing entities? Or predicting wrong labels?
    df_errors = pd.DataFrame(error_data, columns=["Entity", "ErrorType", "Count"])
    
    # We use a custom Vega-Lite chart for grouped bars as it's cleaner in WandB
    # However, a simple approach is logging the table and using WandB's custom chart builder.
    # Here is a code-based chart generation:
    
    table_errors = wandb.Table(dataframe=df_errors)
    
    # This creates a grouped bar chart
    wandb.log({
        "chart_error_breakdown": wandb.plot.bar(
            table_errors, "Entity", "Count", split="ErrorType",
            title="Error Breakdown per Entity (Correct vs Errors)"
        )
    })
    
    print("--- Visualizations Logged to WandB ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate NER models (GLiNER or SpanMarker) and log to WandB.')
    
    parser.add_argument("--model_type", type=str, required=True, choices=["GLINER", "SPANMARKER"])
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved checkpoint")
    
    # WandB specific arguments
    parser.add_argument("--wandb_project", type=str, default="ner-evaluation", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB username/org")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional name for the run")

    parser.add_argument("--wandb_run_id", type=str, help="WandB Run ID to resume")
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
        id=args.wandb_run_id,  # <--- USE PASSED ID
        resume="allow", 
        config=vars(args)
    )

    # 3. Load Data
    print(f"Loading dataset: {args.dataset_id}")
    dataset = load_dataset(args.dataset_id)
    
    # Extract labels (ground truth schema)
    # Assumes "ner_tags" features exist as per your training script
    BIO_label_list = dataset["train"].features["ner_tags"].feature.names
    label_list = list(CLIRENER_LABELS_V1)

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
    results, results_by_tag = run_nervaluate(true_ids, pred_ids, BIO_label_list, tags=label_list)

    # 7. Log
    log_to_wandb(results, results_by_tag)
    
    wandb.finish()