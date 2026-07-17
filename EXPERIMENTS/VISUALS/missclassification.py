import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate detailed SemEval metrics from WandB export to compare Class Confusion vs. Detection failures across all models with standard deviations."
    )
    parser.add_argument(
        "--input_file", type=str, default=None, 
        help="Path to the detailed results CSV/TSV file (e.g., clirener_GOLDagg_detailed_results.csv)"
    )
    args = parser.parse_args()

    # Determine input file path
    input_file = args.input_file
    if input_file is None:
        # Search for typical file patterns in the local directory
        potential_names = [
            "clirener_GOLD_detailed_results_16726",
            "clirener_GOLD_detailed_results_16726.csv",
            "clirener_GOLDagg_detailed_results.csv"
        ]
        
        # Scan current directory for files containing 'detailed_results' or 'GOLD'
        for f in os.listdir("."):
            if f.endswith(".csv") or f.endswith(".tsv") or "detailed_results" in f:
                if f not in potential_names:
                    potential_names.insert(0, f)
                    
        for name in potential_names:
            if os.path.exists(name):
                input_file = name
                break

    if input_file is None or not os.path.exists(input_file):
        print("❌ Error: Could not automatically locate the detailed results file.")
        print("Please run the script and specify your file directly:")
        print("python aggregate_misclassifications.py --input_file <path_to_file>")
        return

    print(f"📖 Reading detailed evaluation data from: {input_file}")
    
    # Auto-detect delimiter (comma vs. tab) to handle raw copy-pastes or CSVs smoothly
    try:
        df = pd.read_csv(input_file, sep=None, engine='python')
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    # Filter out gemma_4_31b_it_zs (case-insensitive and whitespace stripped)
    if "model_display_name" in df.columns:
        df = df[df["model_display_name"].str.lower().str.strip() != "gemma_4_31b_it_zs"]

    # Map required columns for the SemEval Task 9.1 ent_type evaluation
    req_cols = [
        "model_display_name",
        "tag",
        "ent_type_count_correct",
        "ent_type_count_incorrect",
        "ent_type_count_missed",
        "ent_type_count_spurious"
    ]

    missing_cols = [col for col in req_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: The file is missing the following required columns: {missing_cols}")
        print("Please ensure your CSV export contains detailed per-tag 'ent_type' metrics.")
        return

    # Clean and cast columns to numeric, converting nulls/non-numeric values to 0
    for col in req_cols[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # 1. Reconstruct Seed/Run Indices dynamically using cumcount
    df["seed_idx"] = df.groupby(["model_display_name", "tag"]).cumcount()

    # 2. Aggregate counts across all tags to get totals per individual seed/run
    seed_level = df.groupby(["model_display_name", "seed_idx"])[req_cols[2:]].sum().reset_index()

    # 3. Calculate derived metrics at the seed level
    seed_level["total_errors"] = (
        seed_level["ent_type_count_incorrect"] +
        seed_level["ent_type_count_missed"] +
        seed_level["ent_type_count_spurious"]
    )
    
    seed_level["detection_errors"] = (
        seed_level["ent_type_count_missed"] +
        seed_level["ent_type_count_spurious"]
    )

    # Safe division to prevent division-by-zero errors
    divisor = seed_level["total_errors"].replace(0, 1)
    seed_level["confusion_pct"] = (seed_level["ent_type_count_incorrect"] / divisor) * 100
    seed_level["detection_pct"] = (seed_level["detection_errors"] / divisor) * 100
    
    # Reset percentages to zero if there are no errors at all
    seed_level.loc[seed_level["total_errors"] == 0, ["confusion_pct", "detection_pct"]] = 0.0

    # 4. Calculate Mean and Standard Deviation across seeds per model
    aggregated = seed_level.groupby("model_display_name").agg({
        "ent_type_count_correct": ["mean", "std"],
        "ent_type_count_incorrect": ["mean", "std"],
        "ent_type_count_missed": ["mean", "std"],
        "ent_type_count_spurious": ["mean", "std"],
        "detection_errors": ["mean", "std"],
        "total_errors": ["mean", "std"],
        "confusion_pct": ["mean", "std"],
        "detection_pct": ["mean", "std"]
    })

    # Flatten column MultiIndex (e.g., 'ent_type_count_correct_mean')
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
    aggregated = aggregated.reset_index()

    # Handle standard deviation for single-run/zero-shot models (std is NaN -> 0.0)
    aggregated = aggregated.fillna(0.0)

    # Sort models by correct count mean descending
    aggregated = aggregated.sort_values(by="ent_type_count_correct_mean", ascending=False)

    # Print summary table formatted with standard deviations
    print("\n" + "="*172)
    print(f"{'Model Name':<32} | {'Correct (mean ± std)':<22} | {'Confusion (mean ± std)':<32} | {'Detection (mean ± std)':<32} | {'FN (mean ± std)':<16} | {'FP (mean ± std)':<16}")
    print("="*172)
    
    for _, row in aggregated.iterrows():
        name = row["model_display_name"]
        
        correct_str = f"{row['ent_type_count_correct_mean']:.1f} ± {row['ent_type_count_correct_std']:.1f}"
        confusion_str = f"{row['ent_type_count_incorrect_mean']:.1f} ± {row['ent_type_count_incorrect_std']:.1f} ({row['confusion_pct_mean']:.1f}% ± {row['confusion_pct_std']:.1f}%)"
        detection_str = f"{row['detection_errors_mean']:.1f} ± {row['detection_errors_std']:.1f} ({row['detection_pct_mean']:.1f}% ± {row['detection_pct_std']:.1f}%)"
        fn_str = f"{row['ent_type_count_missed_mean']:.1f} ± {row['ent_type_count_missed_std']:.1f}"
        fp_str = f"{row['ent_type_count_spurious_mean']:.1f} ± {row['ent_type_count_spurious_std']:.1f}"
        
        print(f"{name:<32} | "
              f"{correct_str:<22} | "
              f"{confusion_str:<32} | "
              f"{detection_str:<32} | "
              f"{fn_str:<16} | "
              f"{fp_str:<16}")
              
    print("="*172)

    # Save to CSV for easy charting/reporting
    output_summary = "clirener_aggregated_misclassifications.csv"
    aggregated.to_csv(output_summary, index=False)
    print(f"\n✅ Aggregation complete. Summary with spread metrics saved to: {os.path.abspath(output_summary)}")

if __name__ == "__main__":
    main()