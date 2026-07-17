import pandas as pd
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict

def get_gold_counts():
    """Loads the merged Gold dataset and counts the exact frequency of each entity type."""
    print("Fetching actual Gold entity frequencies...")
    ds_gold = load_dataset("P0L3/CliReNER_v_1_1_28_GOLD")
    merged_gold = concatenate_datasets([ds_gold[split] for split in ds_gold.keys()])
    
    labels = merged_gold.features["ner_tags"].feature.names
    
    counts = defaultdict(int)
    for tags in merged_gold['ner_tags']:
        for tag_id in tags:
            tag_name = labels[tag_id]
            if tag_name.startswith("B-"):
                counts[tag_name[2:]] += 1
                
    return counts

def main():
    report_path = "RESULTS/STORED_PREDICTIONS/detailed_class_confusion_report.csv"
    try:
        df = pd.read_csv(report_path)
    except FileNotFoundError:
        print(f"❌ Could not find {report_path}. Run the previous script first.")
        return

    # 1. Get exact gold frequencies
    gold_counts = get_gold_counts()

    # 2. Filter for STRICT inter-class confusions (Ignore Missed 'O' and Spurious 'O')
    df_conf = df[(df["Gold_Label"] != "O") & (df["Predicted_Label"] != "O")].copy()

    # 3. Normalize: What % of the time was Gold misclassified as Pred?
    df_conf["Gold_Total"] = df_conf["Gold_Label"].map(gold_counts)
    df_conf["Confusion_Rate_%"] = (df_conf["Count"] / df_conf["Gold_Total"]) * 100

    # Sort by the highest normalized confusion rate
    df_conf = df_conf.sort_values(by=["Model", "Confusion_Rate_%"], ascending=[True, False])

    # 4. Print Top 5 for each model
    print("\n" + "="*90)
    print("🚨 TOP 5 NORMALIZED INTER-CLASS CONFUSIONS PER MODEL")
    print("="*90)

    for model, group in df_conf.groupby("Model", sort=False): # maintain alphabetical or previous sort
        print(f"\n🟩 MODEL: {model}")
        print(f"{'Gold Label':<25} -> {'Predicted Label':<25} | {'Rate':<7} | {'Errors / Total Gold'}")
        print("-" * 90)
        
        top_5 = group.nlargest(5, "Confusion_Rate_%")
        for _, row in top_5.iterrows():
            g_lbl = row['Gold_Label']
            p_lbl = row['Predicted_Label']
            rate = row['Confusion_Rate_%']
            cnt = int(row['Count'])
            tot = int(row['Gold_Total'])
            
            print(f"{g_lbl:<25} -> {p_lbl:<25} | {rate:>5.1f}%  | {cnt} / {tot}")

    # Save normalized data for later
    out_path = "RESULTS/STORED_PREDICTIONS/normalized_class_confusions.csv"
    df_conf.to_csv(out_path, index=False)
    print(f"\n✅ Normalized confusions saved to: {out_path}")

if __name__ == "__main__":
    main()