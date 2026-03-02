import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datasets import load_dataset, concatenate_datasets
# Ensure dataset_processing.py is in the same directory
from dataset_processing import shorten_name

def get_dataset_counts(dataset_name):
    """
    Loads a dataset, concatenates all splits, and counts entity classes.
    Returns: (Counter object, Safe Name string)
    """
    print(f"Loading {dataset_name}...")
    try:
        ds = load_dataset(dataset_name)
        
        if "train" in ds:
            features = ds["train"].features["ner_tags"].feature
        else:
            first_split = list(ds.keys())[0]
            features = ds[first_split].features["ner_tags"].feature
        
        label_names = features.names
        all_splits = [ds[k] for k in ds.keys()]
        combined_ds = concatenate_datasets(all_splits)
        
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None, None

    class_counts = Counter()
    print(f"Processing tags for {dataset_name}...")
    for row in combined_ds:
        tags = row['ner_tags']
        for tag_id in tags:
            if tag_id < 0 or tag_id >= len(label_names):
                continue
            
            tag_name = label_names[tag_id]
            if tag_name.startswith("B-"):
                class_type = tag_name[2:]
                class_counts[class_type] += 1

    safe_name = shorten_name(dataset_name)
    return class_counts, safe_name

def plot_comparative_distribution(datasets_info, output_dir):
    """
    Plots a grouped bar chart for 2 or 3 datasets.
    datasets_info: List of tuples [(counts, name), ...]
    """
    if not datasets_info:
        return

    # 1. Prepare Data
    data_dict = {name: counts for counts, name in datasets_info}
    df = pd.DataFrame.from_dict(data_dict, orient='index').fillna(0).T
    df = df.sort_index()

    # Calculate Percentages for each dataset
    names = [info[1] for info in datasets_info]
    for name in names:
        total = df[name].sum()
        df[f'{name}_pct'] = (df[name] / total) * 100

    # 2. Plotting Setup
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    
    labels = df.index
    x = np.arange(len(labels))
    num_datasets = len(datasets_info)
    
    # Adjust bar width based on number of datasets
    total_width = 0.8
    bar_width = total_width / num_datasets
    
    colors = ['#8C2981', '#FE9F6D', '#21918C'] # Added a third teal color
    
    fig, ax = plt.subplots(figsize=(14, 7))

    # 3. Plot Bars Dynamically
    for i, name in enumerate(names):
        # Calculate offset: centers the group of bars over the tick
        offset = (i - (num_datasets - 1) / 2) * bar_width
        rects = ax.bar(x + offset, df[f'{name}_pct'], width=bar_width, 
                        label=name, color=colors[i % len(colors)], edgecolor='black')
        
        # Annotation Logic (Raw Counts)
        raw_counts = df[name]
        for rect, count in zip(rects, raw_counts):
            height = rect.get_height()
            ax.annotate(f'{int(count)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # 4. Styling
    ax.set_ylabel('Percentage (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    max_height = df[[f'{n}_pct' for n in names]].values.max()
    ax.set_ylim([0, max_height * 1.15]) 

    plt.tight_layout()

    # 5. Save
    comp_name = "_vs_".join(names)
    filename = f"COMPARE_{comp_name}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Comparison plot saved to: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Compare Class Distributions of HF Datasets")
    parser.add_argument("--d1", type=str, required=True, help="First Dataset HF Name")
    parser.add_argument("--d2", type=str, required=True, help="Second Dataset HF Name")
    parser.add_argument("--d3", type=str, default=None, help="Third Dataset HF Name (Optional)")
    parser.add_argument("--output_dir", type=str, default="PLOTS", help="Output directory")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect data for all provided datasets
    datasets_to_plot = []
    
    # Process D1 and D2
    for d_name in [args.d1, args.d2]:
        counts, safe_name = get_dataset_counts(d_name)
        if counts:
            datasets_to_plot.append((counts, safe_name))
            
    # Process D3 if provided
    if args.d3:
        counts3, safe_name3 = get_dataset_counts(args.d3)
        if counts3:
            datasets_to_plot.append((counts3, safe_name3))

    # Run plotting if we have at least the original two
    if len(datasets_to_plot) >= 2:
        plot_comparative_distribution(datasets_to_plot, args.output_dir)
    else:
        print("Not enough datasets loaded successfully to create a comparison.")

if __name__ == "__main__":
    main()