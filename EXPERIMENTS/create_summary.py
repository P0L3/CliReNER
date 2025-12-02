import wandb
import pandas as pd
import json
import re
import os
import tempfile
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_PROJECT = "CLIRENER_GOLD_SEEDS_authorannots"
TARGET_PROJECT = "CLIRENER_GOLD_SEEDS_authorannots"
REPORT_RUN_NAME = "Final_Report_Multiple_Leaderboards"
METRIC_COL = "strict_f1" 

# Clean names mapping
MODEL_NAME_MAP = {
    'roberta_base': 'RoBERTa Base',
    'bert_base_uncased': 'BERT Base',
    'EnvironmentalBERT_base': 'EnvironmentalBERT',
    'scibert_scivocab_uncased': 'SciBERT',
    'cliscibert_scivocab_uncased': 'CliSciBERT',
    'clirebert_clirevocab_uncased': 'CliReBERT',
    'distilroberta_base': 'RoBERTa (Distil)',
    'distilroberta_base_climate_f': 'ClimateBERT',
    'sciclimatebert': 'SciClimateBERT',
    'gliner_medium_v2_5': 'GLiNER: Medium v2.5',
    'gliner_small_v2_5': 'GLiNER: Small v2.5'
}

# --- GROUPS DEFINITION ---
# 3. BERT Based Models
GROUP_BERT = [
    'SciBERT', 
    'EnvironmentalBERT', 
    'BERT Base', 
    'CliReBERT', 
    'CliSciBERT'
]

# 4. Distil/Climate Models
GROUP_DISTIL = [
    'RoBERTa (Distil)', 
    'ClimateBERT', 
    'SciClimateBERT'
]

def clean_model_name(run_name):
    name = re.sub(r'(_s\d+|_seed\d+)$', '', run_name)
    name = name.replace("eval_GOLD_", "")
    return MODEL_NAME_MAP.get(name, name)

def fetch_and_aggregate():
    print(f"--- 1. Fetching Runs from {SOURCE_PROJECT} ---")
    api = wandb.Api()
    runs = api.runs(path=SOURCE_PROJECT)
    
    all_rows = []
    for run in tqdm(runs, desc="Processing Runs"):
        if run.state != "finished": continue
        clean_name = clean_model_name(run.name)
        
        artifacts = [a for a in run.logged_artifacts() if "run_table" in a.type]
        target_artifact = next((a for a in artifacts if "per_tag_results" in a.name), None)
        
        if target_artifact:
            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    dir_path = target_artifact.download(root=tmp_dir) + "/tables/"
                    json_file = [f for f in os.listdir(dir_path) if f.endswith(".json")][0]
                    with open(os.path.join(dir_path, json_file), 'r') as f:
                        table_dict = json.load(f)
                    
                    df_run = pd.DataFrame(table_dict["data"], columns=table_dict["columns"])
                    df_run["Model"] = clean_name
                    df_run["Seed"] = run.config.get("seed", 0)
                    
                    keep_cols = ["tag", "Model", METRIC_COL]
                    all_rows.append(df_run[keep_cols])
                except Exception as e:
                    print(e)
                    pass
    if not all_rows: return None
    return pd.concat(all_rows, ignore_index=True)

def generate_leaderboard(stats_df, title_suffix="", allowed_models=None, excluded_substring=None):
    """
    Generic function to generate a Top 3 table based on filters.
    """
    df = stats_df.copy()
    
    # Apply Inclusion Filter
    if allowed_models:
        df = df[df["Model"].isin(allowed_models)]
        
    # Apply Exclusion Filter
    if excluded_substring:
        df = df[~df["Model"].str.contains(excluded_substring)]
        
    unique_tags = sorted(df["tag"].unique())
    leaderboard_rows = []
    
    for tag in unique_tags:
        tag_data = df[df["tag"] == tag].sort_values(by="mean", ascending=False)
        top3 = tag_data.head(3).reset_index(drop=True)
        
        row = {"Entity Class": tag}
        for i in range(3):
            if i < len(top3):
                m_name = top3.iloc[i]["Model"]
                score = top3.iloc[i]["mean"]
                std = top3.iloc[i]["std"]
                row[f"Rank {i+1}"] = f"{m_name} ({score:.3f} ±{std:.3f})"
            else:
                row[f"Rank {i+1}"] = "-"
        leaderboard_rows.append(row)
        
    return pd.DataFrame(leaderboard_rows)

def process_and_upload(raw_df):
    print(f"--- 2. Calculating Statistics ---")
    
    # Calculate Mean/Std once for the whole dataset
    full_stats = raw_df.groupby(["tag", "Model"])[METRIC_COL].agg(['mean', 'std']).reset_index()
    
    print(f"--- 3. Generating Tables ---")
    
    # 1. ALL MODELS
    df_all = generate_leaderboard(full_stats)
    
    # 2. NO GLINER
    df_no_gliner = generate_leaderboard(full_stats, excluded_substring="GLiNER")
    
    # 3. BERT BASED ONLY
    df_bert = generate_leaderboard(full_stats, allowed_models=GROUP_BERT)
    
    # 4. DISTIL BASED ONLY
    df_distil = generate_leaderboard(full_stats, allowed_models=GROUP_DISTIL)

    print(f"--- 4. Uploading to New Run: {REPORT_RUN_NAME} ---")
    
    run = wandb.init(
        project=TARGET_PROJECT,
        name=REPORT_RUN_NAME,
        job_type="report_generation",
        config={"description": "Multi-table leaderboard summary"}
    )
    
    # Log tables with specific keys
    run.log({
        "1_Leaderboard_ALL_MODELS": wandb.Table(dataframe=df_all),
        "2_Leaderboard_NO_GLINER": wandb.Table(dataframe=df_no_gliner),
        "3_Leaderboard_BERT_ONLY": wandb.Table(dataframe=df_bert),
        "4_Leaderboard_DISTIL_ONLY": wandb.Table(dataframe=df_distil),
        "Raw_Stats": wandb.Table(dataframe=full_stats) # Backup
    })
    
    print("Upload Complete.")
    print(f"View Run here: {run.get_url()}")
    run.finish()

if __name__ == "__main__":
    raw_df = fetch_and_aggregate()
    if raw_df is not None:
        process_and_upload(raw_df)