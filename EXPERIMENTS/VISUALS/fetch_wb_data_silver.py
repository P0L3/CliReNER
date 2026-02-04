import wandb
import pandas as pd
import re
import os
import json
import shutil

# 1. Configuration
ENTITY = "andrija-2"
PROJECT = "CLIRENER_SILVER_SEEDS"  # Updated Project Name
OUTPUT_FILE = "clirener_silver_detailed_results.csv"

# 2. Define the Mapping Dictionary (Extracted from Vega-Lite Spec)
MODEL_NAME_MAP = {
    'RoBERTa': 'RoBERTa Base',
    'BERT': 'BERT Base',
    'EnvironmentalBERT': 'EnvironmentalBERT',
    'scibert_scivocab_uncased': 'SciBERT',
    'cliscibert_scivocab_uncased': 'CliSciBERT',
    'clirebert_clirevocab_uncased': 'CliReBERT',
    'DistilRoBERTa': 'Distil RoBERTa',
    'distilroberta_base_climate_f': 'ClimateBERT',
    'sciclimatebert': 'SciClimateBERT',
    'gliner_medium_v2_5': 'GLiNER: Medium v2.5',
    'gliner_small_v2_5': 'GLiNER: Small v2.5',
    'ZS_gliner_medium_v2_5': 'GLiNER: Medium v2.5 ZS',
    'ZS_gliner_small_v2_5': 'GLiNER: Small v2.5 ZS',
    'gpt_5_2_pro_zs': 'GPT 5.2 Pro ZS',
    'gemini_2_5_pro_zs': 'Gemini 2.5 Pro ZS',
    'gpt_5_1_zs': 'GPT 5.1 ZS',
    'gemini_3_pro_preview_zs': 'Gemini 3.0 Pro ZS',
    'deepseek_reasoner_zs': 'DeepSeek-V3.2 (Thinking) ZS',
    'deepseek_chat_zs': 'DeepSeek-V3.2 (Non-Thinking) ZS',
    'claude_sonnet_4_5_zs': 'Claude Sonnet 4.5 ZS',
    'claude_opus_4_5_zs': 'Claude Opus 4.5 ZS',
    'INDUSbase': 'INDUS Base',
    'INDUSbaseSDE': 'INDUS SDE Base'
}

def clean_model_name(run_name):
    """
    Based on Vega transform: 
    replace(replace(datum.name, /_seed\\d+$/, ''), 'SpanMarker_', '')
    """
    # 1. Remove '_seed' followed by digits at the very end of the string
    raw_key = re.sub(r'_seed\d+$', '', run_name)
    
    # 2. Remove 'SpanMarker_' prefix if present
    raw_key = raw_key.replace('SpanMarker_', '')
    
    return raw_key

def main():
    print(f"Connecting to W&B project: {ENTITY}/{PROJECT}...")
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    all_data_frames = []
    
    print(f"Found {len(runs)} runs. Processing...")

    for run in runs:
        # --- 1. Prepare Metadata ---
        raw_key = clean_model_name(run.name)
        display_name = MODEL_NAME_MAP.get(raw_key, raw_key)
        
        summary = run.summary
        
        # Scalar metrics (Overall)
        overall_stats = {
            "model_display_name": display_name,
            "overall_strict_f1": summary.get("overall/strict_f1"),
            "overall_exact_f1": summary.get("overall/exact_f1"),
            "overall_partial_f1": summary.get("overall/partial_f1"),
            "overall_type_f1": summary.get("overall/type_f1"),
        }

        # --- 2. Extract Table ---
        # We look specifically for the table key
        table_key = "tables/per_tag_results"
        
        if table_key in summary:
            table_def = summary[table_key]
            
            # Check if we have a file path (Prioritize this!)
            file_path = None
            if hasattr(table_def, 'get'):
                file_path = table_def.get("path")
            
            if file_path:
                try:
                    # Download the specific JSON file for this table
                    downloaded_file = run.file(file_path).download(replace=True, root=".")
                    
                    # The file downloads to ./media/table/tables/filename.json
                    local_full_path = downloaded_file.name
                    
                    with open(local_full_path, 'r') as f:
                        table_json = json.load(f)
                    
                    # W&B Tables are stored as JSON: {"columns": [...], "data": [[...]]}
                    df_tags = pd.DataFrame(data=table_json["data"], columns=table_json["columns"])
                    
                    # Merge Overall Stats into every row
                    for k, v in overall_stats.items():
                        df_tags[k] = v
                        
                    all_data_frames.append(df_tags)
                    print(f"  [OK] {display_name} (Raw: {raw_key})")
                    
                    # Cleanup: Delete the specific file we just downloaded
                    try:
                        os.remove(local_full_path)
                    except:
                        pass
                        
                except Exception as e:
                    print(f"  [ERR] {display_name}: Failed to download/parse file '{file_path}'. {e}")
            else:
                print(f"  [SKIP] {display_name}: No 'path' key found in table summary.")
        else:
            print(f"  [SKIP] {display_name}: Table '{table_key}' not found.")

    # --- 3. Save to CSV ---
    if all_data_frames:
        print("\nConcatenating...")
        final_df = pd.concat(all_data_frames, ignore_index=True)
        
        # Reorder columns for convenience
        cols = list(final_df.columns)
        desired_order = ['model_display_name', 'tag', 'metric_tag', 'overall_strict_f1']
        
        # Move desired columns to front if they exist
        for col in reversed(desired_order):
            if col in cols:
                cols.insert(0, cols.pop(cols.index(col)))
        
        final_df = final_df[cols]
        final_df.to_csv(OUTPUT_FILE, index=False)
        
        # Final cleanup of media folder created by downloads
        if os.path.exists("media"):
            shutil.rmtree("media", ignore_errors=True)
            
        print(f"SUCCESS: Data saved to {os.path.abspath(OUTPUT_FILE)}")
    else:
        print("No data extracted.")

if __name__ == "__main__":
    main()