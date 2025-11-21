"""
NER Dataset Statistics & Visualization Tool

Author: Andrija Poleksić

Description:
    This script calculates statistics (sentence length, entity density, span length) 
    and generates class distribution plots for Hugging Face NER datasets. It supports 
    train/validation/test splits as well as an aggregated 'overall' view.

Disclaimer:
    This code was reformatted, optimized, and extended with additional features 
    (CLI support, visualization, CSV export) by an AI Assistant.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset, concatenate_datasets

# --- Helper Function to Calculate Stats ---
def process_dataset_split(dataset_split, label_names):
    """
    Parses a dataset split to return raw statistics and class counts.
    """
    sentence_lengths = []
    entity_densities = [] 
    span_lengths = []
    class_counts = Counter()

    # Pre-check: if dataset is empty
    if len(dataset_split) == 0:
        return None, None

    for row in dataset_split:
        tokens = row['tokens']
        tags = row['ner_tags']
        
        # 1. Sentence Length
        sentence_lengths.append(len(tokens))
        
        current_sentence_entity_count = 0
        current_span_length = 0
        
        for tag_id in tags:
            # Handle potential index errors if tags don't align with features
            if tag_id < 0 or tag_id >= len(label_names):
                continue
                
            tag_name = label_names[tag_id]
            
            if tag_name.startswith("B-"):
                # Previous span ended
                if current_span_length > 0:
                    span_lengths.append(current_span_length)
                
                # Start new span
                current_span_length = 1
                current_sentence_entity_count += 1
                
                # Log the class (remove "B-")
                class_type = tag_name[2:]
                class_counts[class_type] += 1
                
            elif tag_name.startswith("I-") and current_span_length > 0:
                current_span_length += 1
            
            else: # Tag is "O"
                if current_span_length > 0:
                    span_lengths.append(current_span_length)
                    current_span_length = 0

        # Capture span if sentence ends while inside an entity
        if current_span_length > 0:
            span_lengths.append(current_span_length)
            
        entity_densities.append(current_sentence_entity_count)

    stats = {
        "Sentence Length (Mean)": np.mean(sentence_lengths),
        "Sentence Length (Median)": np.median(sentence_lengths),
        "Entity Density (Avg entities/sent)": np.mean(entity_densities),
        "Span Length (Avg tokens/entity)": np.mean(span_lengths) if span_lengths else 0,
        "Total Entities": sum(entity_densities),
        "Total Sentences": len(sentence_lengths)
    }
    
    return stats, class_counts

# --- Helper Function to Plot ---
def plot_class_distribution(class_counts, split_name, hf_name, safe_hf_name, output_dir):
    """
    Generates and saves a histogram of class distribution.
    """
    if not class_counts:
        return

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Count'])
    
    # Sort Alphabetically by Index (Entity Name)
    df = df.sort_index()

    # Plotting
    plt.figure(figsize=(12, 6)) 
    bars = plt.bar(df.index, df['Count'], color='skyblue', edgecolor='black')
    
    plt.title(f"Class Distribution - {split_name.upper()}\n({hf_name})")
    plt.xlabel("Entity Type")
    plt.ylabel("Count")
    plt.xticks(rotation=90) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add numbers on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), 
                 va='bottom', ha='center', fontsize=8, rotation=90)

    plt.tight_layout()
    
    # Save File
    filename = f"{safe_hf_name}_{split_name}_class_dist.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close() 

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Calculate and Plot NER Statistics from HF Dataset")
    parser.add_argument("--dataset", type=str, default="P0L3/CliReNER_v_1_1_28_SILVER", 
                        help="Hugging Face dataset path (default: P0L3/CliReNER_v_1_1_28_SILVER)")
    parser.add_argument("--output_dir", type=str, default="PLOTS", 
                        help="Directory to save plots and CSV (default: PLOTS)")
    
    args = parser.parse_args()
    
    hf_name = args.dataset
    output_dir = args.output_dir
    
    # Sanitize model name for filenames
    safe_hf_name = hf_name.replace("/", "_").replace("\\", "_")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset: {hf_name}...")
    try:
        hf_dataset = load_dataset(hf_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- 2. Data Preparation ---
    # Get Label Names from the 'train' split
    if "train" in hf_dataset:
        tags_feature = hf_dataset["train"].features["ner_tags"].feature
    else:
        # Fallback if train doesn't exist, check other splits
        available_split = list(hf_dataset.keys())[0]
        tags_feature = hf_dataset[available_split].features["ner_tags"].feature
        
    label_names = tags_feature.names

    # Create Overall Dataset
    datasets_list = [hf_dataset[split] for split in hf_dataset.keys()]
    combined_dataset = concatenate_datasets(datasets_list)

    # Define splits dictionary
    splits_to_process = {}
    for split in hf_dataset.keys():
        splits_to_process[split] = hf_dataset[split]
    splits_to_process["overall"] = combined_dataset

    # --- 3. Calculation Loop ---
    print(f"\n{'Split':<12} | {'Sent Len':<10} | {'Density':<10} | {'Span Len':<10} | {'Entities':<10}")
    print("-" * 65)

    all_stats_data = []

    for split_name, dataset_obj in splits_to_process.items():
        # Calculate Stats
        stats, counts = process_dataset_split(dataset_obj, label_names)
        
        if stats is None:
            print(f"Skipping empty split: {split_name}")
            continue

        # Print to Console
        print(f"{split_name:<12} | "
              f"{stats['Sentence Length (Mean)']:<10.2f} | "
              f"{stats['Entity Density (Avg entities/sent)']:<10.2f} | "
              f"{stats['Span Length (Avg tokens/entity)']:<10.2f} | "
              f"{int(stats['Total Entities']):<10}")
        
        # Generate Plot
        plot_class_distribution(counts, split_name, hf_name, safe_hf_name, output_dir)
        
        # Collect Data for CSV
        # Add the split name to the dictionary for the CSV row
        csv_row = {"Split": split_name}
        csv_row.update(stats)
        all_stats_data.append(csv_row)

    print("-" * 65)

    # --- 4. Save to CSV ---
    csv_filename = f"{safe_hf_name}_statistics.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    
    df_stats = pd.DataFrame(all_stats_data)
    
    # Reorder columns to put Split first (just in case dict order varies)
    cols = ["Split", "Sentence Length (Mean)", "Sentence Length (Median)", 
            "Entity Density (Avg entities/sent)", "Span Length (Avg tokens/entity)", 
            "Total Entities", "Total Sentences"]
    df_stats = df_stats[cols]
    
    df_stats.to_csv(csv_filepath, index=False)
    print(f"\nStatistics saved to CSV: {csv_filepath}")
    print(f"Plots saved to directory: {output_dir}/")

if __name__ == "__main__":
    main()