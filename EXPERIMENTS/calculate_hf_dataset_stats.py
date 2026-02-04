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

    # Calculate Aggregates
    mean_len = np.mean(sentence_lengths)
    median_len = np.median(sentence_lengths)
    avg_density = np.mean(entity_densities)
    avg_span = np.mean(span_lengths) if span_lengths else 0
    
    # Calculate Ratios
    # Ratio 1: Median Length / Density (Robust)
    ratio_median = median_len / avg_density if avg_density > 0 else 0
    
    # Ratio 2: Mean Length / Density (Token Budgeting)
    ratio_mean = mean_len / avg_density if avg_density > 0 else 0

    stats = {
        "Sentence Length (Mean)": mean_len,
        "Sentence Length (Median)": median_len,
        "Entity Density (Avg entities/sent)": avg_density,
        "Ratio (Median Len / Density)": ratio_median,
        "Ratio (Mean Len / Density)": ratio_mean,
        "Span Length (Avg tokens/entity)": avg_span,
        "Total Entities": sum(entity_densities),
        "Total Sentences": len(sentence_lengths)
    }
    
    return stats, class_counts

def plot_class_distribution(class_counts, split_name, hf_name, safe_hf_name, output_dir, normalize=False, count_label=True):
    """
    Generates and saves a standard bar chart.
    
    Args:
        normalize (bool): If True, bar height represents percentage (0-100).
        count_label (bool): If True, the text on top of bars shows the raw count (int),
                            retrieved directly from source data.
    """
    if not class_counts:
        return

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Count'])
    
    # Sort Alphabetically by Index (Entity Name) BEFORE plotting to ensure alignment
    df = df.sort_index()

    # Determine values to plot (Y-axis height)
    total = df['Count'].sum()
    if normalize and total > 0:
        plot_values = (df['Count'] / total) * 100
    else:
        plot_values = df['Count']

    # Plotting
    plt.figure(figsize=(12, 6)) 
    
    # Standard Vertical Bars
    bars = plt.bar(df.index, plot_values, color='#e05206', edgecolor='black')
    
    # X-Axis Styling
    plt.xticks(rotation=45, ha='right') 
    
    # Y-Axis Bounds
    if normalize:
        plt.ylim([0, 20])
    else:
        plt.ylim([0, 360])
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Access raw counts directly for annotation
    # Since df is sorted, df['Count'] aligns 1:1 with 'bars'
    raw_counts = df['Count'].values

    for bar, raw_count in zip(bars, raw_counts):
        y_plot_val = bar.get_height() # This is the visual height (%, or count)
        
        # Determine what text to write
        if count_label:
            # Direct access to the raw integer (no reconstruction math)
            label_text = str(int(raw_count))
        else:
            # If not using count labels, display the formatted Y-value
            if normalize:
                label_text = f"{y_plot_val:.1f}%"
            else:
                label_text = str(int(y_plot_val))

        plt.text(bar.get_x() + bar.get_width()/2, 
                 y_plot_val, 
                 label_text, 
                 va='bottom', 
                 ha='center', 
                 fontsize=9)

    plt.tight_layout()
    
    # Save File
    suffix = "_norm" if normalize else ""
    filename = f"{safe_hf_name}_{split_name}_class_dist{suffix}.png"
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
    parser.add_argument("--norm",  action=argparse.BooleanOptionalAction, default=False,
                        help="Normalize the output?")
    
    args = parser.parse_args()
    
    normalize = args.norm

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
    # Updated Header to show both ratios
    header = f"{'Split':<12} | {'Len(Mean)':<10} | {'Density':<10} | {'Ratio(Med)':<10} | {'Ratio(Mean)':<10} | {'Span Len':<10} | {'Entities':<10}"
    print("\n" + header)
    print("-" * len(header))

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
              f"{stats['Ratio (Median Len / Density)']:<10.2f} | "
              f"{stats['Ratio (Mean Len / Density)']:<10.2f} | "
              f"{stats['Span Length (Avg tokens/entity)']:<10.2f} | "
              f"{int(stats['Total Entities']):<10}")
        
        # Generate Plot
        plot_class_distribution(counts, split_name, hf_name, safe_hf_name, output_dir, normalize=normalize)
        
        # Collect Data for CSV
        csv_row = {"Split": split_name}
        csv_row.update(stats)
        all_stats_data.append(csv_row)

    print("-" * len(header))

    # --- 4. Save to CSV ---
    csv_filename = f"{safe_hf_name}_statistics.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    
    df_stats = pd.DataFrame(all_stats_data)
    
    # Reorder columns for readability in CSV
    cols = [
        "Split", 
        "Sentence Length (Mean)", 
        "Sentence Length (Median)", 
        "Entity Density (Avg entities/sent)", 
        "Ratio (Median Len / Density)",
        "Ratio (Mean Len / Density)",
        "Span Length (Avg tokens/entity)", 
        "Total Entities", 
        "Total Sentences"
    ]
    
    # Ensure we only select columns that exist
    cols = [c for c in cols if c in df_stats.columns]
    df_stats = df_stats[cols]
    
    df_stats.to_csv(csv_filepath, index=False)
    print(f"\nStatistics saved to CSV: {csv_filepath}")
    print(f"Plots saved to directory: {output_dir}/")

if __name__ == "__main__":
    main()