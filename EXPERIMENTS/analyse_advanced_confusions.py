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
        print(f"❌ Could not find {report_path}. Run the inference script first.")
        return

    # 1. Prepare Data
    gold_counts = get_gold_counts()
    
    # Filter for STRICT inter-class confusions (Ignore 'O')
    df_conf = df[(df["Gold_Label"] != "O") & (df["Predicted_Label"] != "O")].copy()
    df_conf["Gold_Total"] = df_conf["Gold_Label"].map(gold_counts)

    # Classify models: Zero-Shot vs Fine-Tuned
    df_conf["Is_ZeroShot"] = df_conf["Model"].str.contains("ZS")
    num_zs = df_conf[df_conf["Is_ZeroShot"]]["Model"].nunique()
    num_ft = df_conf[~df_conf["Is_ZeroShot"]]["Model"].nunique()
    num_total = df_conf["Model"].nunique()

    print(f"\nAnalyzed {num_total} models ({num_zs} Zero-Shot, {num_ft} Fine-Tuned).")

    # =========================================================================
    # NARRATIVE 1: GLOBAL CONSENSUS BOTTLENECKS (Taxonomy Ambiguity)
    # =========================================================================
    print("\n" + "="*80)
    print("🌍 NARRATIVE 1: GLOBAL TAXONOMY BOTTLENECKS")
    print("   (Pairs consistently confused by ALL models, indicating guideline ambiguity)")
    print("="*80)
    
    global_agg = df_conf.groupby(["Gold_Label", "Predicted_Label"])["Count"].sum().reset_index()
    global_agg["Gold_Total"] = global_agg["Gold_Label"].map(gold_counts)
    # Normalize by (Total Models * Frequency of Gold Label)
    global_agg["Global_Confusion_Rate_%"] = (global_agg["Count"] / (global_agg["Gold_Total"] * num_total)) * 100
    
    top_global = global_agg.sort_values(by="Global_Confusion_Rate_%", ascending=False).head(10)
    print(f"{'Gold Label':<25} -> {'Predicted Label':<25} | {'Global Rate'}")
    print("-" * 65)
    for _, row in top_global.iterrows():
        print(f"{row['Gold_Label']:<25} -> {row['Predicted_Label']:<25} | {row['Global_Confusion_Rate_%']:>5.1f}%")

    # =========================================================================
    # NARRATIVE 2: ZERO-SHOT VS. FINE-TUNED (Inductive Bias Asymmetry)
    # =========================================================================
    print("\n" + "="*80)
    print("🧠 NARRATIVE 2: ZERO-SHOT PRIORS VS. FINE-TUNED ADAPTATION")
    print("   (Where do pre-trained semantics fail that supervised learning fixes?)")
    print("="*80)

    # Aggregate for Zero-Shot
    zs_agg = df_conf[df_conf["Is_ZeroShot"]].groupby(["Gold_Label", "Predicted_Label"])["Count"].sum().reset_index()
    zs_agg["ZS_Rate_%"] = (zs_agg["Count"] / (zs_agg["Gold_Label"].map(gold_counts) * num_zs)) * 100
    
    # Aggregate for Fine-Tuned
    ft_agg = df_conf[~df_conf["Is_ZeroShot"]].groupby(["Gold_Label", "Predicted_Label"])["Count"].sum().reset_index()
    ft_agg["FT_Rate_%"] = (ft_agg["Count"] / (ft_agg["Gold_Label"].map(gold_counts) * num_ft)) * 100

    # Merge
    bias_df = pd.merge(zs_agg[["Gold_Label", "Predicted_Label", "ZS_Rate_%"]], 
                       ft_agg[["Gold_Label", "Predicted_Label", "FT_Rate_%"]], 
                       on=["Gold_Label", "Predicted_Label"], how="outer").fillna(0)
    
    bias_df["Delta (ZS - FT)"] = bias_df["ZS_Rate_%"] - bias_df["FT_Rate_%"]
    
    top_llm_fails = bias_df.sort_values(by="Delta (ZS - FT)", ascending=False).head(5)
    print("🔹 WORST ZERO-SHOT HALLUCINATIONS (Fixed by Fine-Tuning):")
    print(f"{'Gold Label':<22} -> {'Predicted Label':<22} | {'ZS Rate':<7} | {'FT Rate':<7} | {'Delta'}")
    print("-" * 80)
    for _, row in top_llm_fails.iterrows():
        print(f"{row['Gold_Label']:<22} -> {row['Predicted_Label']:<22} | {row['ZS_Rate_%']:>5.1f}%  | {row['FT_Rate_%']:>5.1f}%  | +{row['Delta (ZS - FT)']:>4.1f}%")

    top_ft_fails = bias_df.sort_values(by="Delta (ZS - FT)", ascending=True).head(5)
    print("\n🔸 WORST FINE-TUNED OVERFITTING (Where Zero-Shot is actually safer):")
    print(f"{'Gold Label':<22} -> {'Predicted Label':<22} | {'ZS Rate':<7} | {'FT Rate':<7} | {'Delta'}")
    print("-" * 80)
    for _, row in top_ft_fails.iterrows():
        print(f"{row['Gold_Label']:<22} -> {row['Predicted_Label']:<22} | {row['ZS_Rate_%']:>5.1f}%  | {row['FT_Rate_%']:>5.1f}%  | {row['Delta (ZS - FT)']:>4.1f}%")

    # =========================================================================
    # NARRATIVE 3: DIRECTIONAL HIERARCHY (Asymmetry of Confusion)
    # =========================================================================
    print("\n" + "="*80)
    print("📐 NARRATIVE 3: IMPLICIT HIERARCHY (Directional Fallback)")
    print("   (Model backs off from specific to broad, but rarely broad to specific)")
    print("="*80)

    # We use global rates to find pairs with high asymmetry
    asym_list = []
    processed_pairs = set()

    for _, row in global_agg.iterrows():
        g_lbl = row["Gold_Label"]
        p_lbl = row["Predicted_Label"]
        rate_A_to_B = row["Global_Confusion_Rate_%"]
        
        # Avoid processing A->B and B->A twice
        pair_key = tuple(sorted([g_lbl, p_lbl]))
        if pair_key in processed_pairs: continue
        processed_pairs.add(pair_key)
        
        # Find reverse rate (B -> A)
        reverse_row = global_agg[(global_agg["Gold_Label"] == p_lbl) & (global_agg["Predicted_Label"] == g_lbl)]
        rate_B_to_A = reverse_row["Global_Confusion_Rate_%"].values[0] if not reverse_row.empty else 0.0
        
        # We care about pairs where one direction is highly common (>5%) but the other is rare
        if max(rate_A_to_B, rate_B_to_A) > 5.0:
            asym_list.append({
                "Entity_1": g_lbl,
                "Entity_2": p_lbl,
                "1_to_2_Rate": rate_A_to_B,
                "2_to_1_Rate": rate_B_to_A,
                "Asymmetry": abs(rate_A_to_B - rate_B_to_A)
            })

    asym_df = pd.DataFrame(asym_list).sort_values(by="Asymmetry", ascending=False).head(5)
    print(f"{'Specific (A)':<22} -> {'Broad (B)':<22} | {'A -> B Rate':<12} | {'B -> A Rate'}")
    print("-" * 80)
    for _, row in asym_df.iterrows():
        # Figure out which is the dominant direction
        if row["1_to_2_Rate"] > row["2_to_1_Rate"]:
            spec, broad = row["Entity_1"], row["Entity_2"]
            r_dom, r_sub = row["1_to_2_Rate"], row["2_to_1_Rate"]
        else:
            spec, broad = row["Entity_2"], row["Entity_1"]
            r_dom, r_sub = row["2_to_1_Rate"], row["1_to_2_Rate"]
            
        print(f"{spec:<22} -> {broad:<22} | {r_dom:>5.1f}%       | {r_sub:>5.1f}%")

if __name__ == "__main__":
    main()