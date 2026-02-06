import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datasets import load_dataset, concatenate_datasets
from dataset_processing import shorten_name

def get_dataset_counts(dataset_name):
    """
    Loads a dataset, concatenates all splits, and counts entity classes.
    Returns: (Counter object, Safe Name string)
    """
    print(f"Loading {dataset_name}...")
    try:
        # Load dataset
        ds = load_dataset(dataset_name)
        
        # Determine label names
        if "train" in ds:
            features = ds["train"].features["ner_tags"].feature
        else:
            first_split = list(ds.keys())[0]
            features = ds[first_split].features["ner_tags"].feature
        
        label_names = features.names
        
        # Concatenate all splits (Train + Val + Test) for holistic view
        all_splits = [ds[k] for k in ds.keys()]
        combined_ds = concatenate_datasets(all_splits)
        
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None, None

    # Count Tags
    class_counts = Counter()
    
    print(f"Processing tags for {dataset_name}...")
    for row in combined_ds:
        tags = row['ner_tags']
        for tag_id in tags:
            if tag_id < 0 or tag_id >= len(label_names):
                continue
            
            tag_name = label_names[tag_id]
            
            # Logic to extract class name (assuming B-XYZ, I-XYZ format)
            if tag_name.startswith("B-"):
                class_type = tag_name[2:]
                class_counts[class_type] += 1
                
            # If you want to count single tokens without BIO logic, adjust here.
            # Currently counts 'Entities' (B- tags), not total tokens.

    safe_name = shorten_name(dataset_name)
    return class_counts, safe_name

def plot_comparative_distribution(counts1, name1, counts2, name2, output_dir):
    """
    Plots a grouped bar chart.
    Height = Percentage (Normalized).
    Text = Raw Count.
    """
    if not counts1 or not counts2:
        print("Error: Missing data for plotting.")
        return

    # 1. Prepare Data
    # Create a DataFrame to align classes (handles missing classes in one ds)
    data = {
        name1: counts1,
        name2: counts2
    }
    df = pd.DataFrame.from_dict(data, orient='index').fillna(0).T
    df = df.sort_index()

    # Calculate Totals
    total1 = df[name1].sum()
    total2 = df[name2].sum()

    # Calculate Percentages
    df[f'{name1}_pct'] = (df[name1] / total1) * 100
    df[f'{name2}_pct'] = (df[name2] / total2) * 100

    # 2. Plotting Setup
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    labels = df.index
    x = np.arange(len(labels))  # Label locations
    bar_width = 0.45

    # The offset from the center 'x' for each bar is half its width.
    offset = bar_width / 2

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot the bars using the calculated offset and the defined bar_width.
    # Bar 1 is shifted left from the center, Bar 2 is shifted right.
    rects1 = ax.bar(x - offset, df[f'{name1}_pct'], width=bar_width, label=name1, color='#8C2981', edgecolor='black')
    rects2 = ax.bar(x + offset, df[f'{name2}_pct'], width=bar_width, label=name2, color='#FE9F6D', edgecolor='black')


    # 3. Styling
    ax.set_ylabel('Percentage (%)')
    # ax.set_title('Dataset Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Dynamic Y-Limit to ensure text fits
    max_height = max(df[f'{name1}_pct'].max(), df[f'{name2}_pct'].max())
    ax.set_ylim([0, max_height * 1.10]) # Add 15% headroom

    # 4. Annotation Logic (Raw Counts)
    def autolabel(rects, raw_counts):
        """Attach a text label above each bar displaying raw count."""
        for rect, count in zip(rects, raw_counts):
            height = rect.get_height()
            ax.annotate(f'{int(count)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=0)

    # Pass the raw counts (Series) to the labeler
    autolabel(rects1, df[name1])
    autolabel(rects2, df[name2])

    plt.tight_layout()

    # 5. Save
    filename = f"COMPARE_{name1}_vs_{name2}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Comparison plot saved to: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Compare Class Distributions of Two HF Datasets")
    parser.add_argument("--d1", type=str, required=True, help="First Dataset HF Name")
    parser.add_argument("--d2", type=str, required=True, help="Second Dataset HF Name")
    parser.add_argument("--output_dir", type=str, default="PLOTS", help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Get Counts
    counts1, safe_name1 = get_dataset_counts(args.d1)
    counts2, safe_name2 = get_dataset_counts(args.d2)

    if counts1 and counts2:
        plot_comparative_distribution(counts1, safe_name1, counts2, safe_name2, args.output_dir)

if __name__ == "__main__":
    main()