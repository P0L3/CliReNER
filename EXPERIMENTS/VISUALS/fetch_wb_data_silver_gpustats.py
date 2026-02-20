import wandb
import pandas as pd
import re
import os

# 1. Configuration
ENTITY = "andrija-2"
PROJECT = "CLIRENER_SILVER_SEEDS"
OUTPUT_FILE = "gpu_power_usage_results.csv"

# 2. Define the Mapping Dictionary
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
    raw_key = re.sub(r'_seed\d+$', '', run_name)
    raw_key = raw_key.replace('SpanMarker_', '')
    return raw_key

def main():
    print(f"Connecting to W&B project: {ENTITY}/{PROJECT}...")
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    results = []
    
    print(f"Found {len(runs)} runs. Processing GPU Power metrics...")

    for run in runs:
        raw_key = clean_model_name(run.name)
        display_name = MODEL_NAME_MAP.get(raw_key, raw_key)
        
        try:
            # Fetch events stream (System metrics)
            try:
                events_df = run.history(stream="events", samples=100000)
            except Exception:
                events_df = pd.DataFrame()
            
            # Fallback for newer W&B versions
            if events_df.empty:
                events_df = run.history(samples=100000)
                
            if events_df.empty:
                print(f"  [SKIP] {display_name}: No history data found.")
                continue

            # --- CORRECTION HERE ---
            # 1. Find columns that contain 'powerWatts' (e.g., system.gpu.0.powerWatts)
            power_cols = [c for c in events_df.columns if "powerWatts" in c]
            
            if not power_cols:
                # This usually happens for API models (GPT, Gemini) or CPU-only runs
                print(f"  [SKIP] {display_name}: No GPU power metrics found.")
                continue
            
            # 2. Filter the DataFrame to ONLY those columns
            # We drop rows where *all* power columns are NaN (e.g., logging gaps)
            power_data = events_df[power_cols].dropna(how='all')
            
            if power_data.empty:
                print(f"  [SKIP] {display_name}: GPU power data is entirely empty/NaN.")
                continue
                
            # 3. Sum power across all GPUs (if multi-GPU) for each timestamp
            total_power_per_step = power_data.sum(axis=1)
            
            # 4. Calculate average power over the duration of the run
            avg_power_w = total_power_per_step.mean()
            
            results.append({
                "model_display_name": display_name,
                "raw_key": raw_key,
                "avg_gpu_power_W": round(avg_power_w, 2),
                "num_gpus_detected": len(power_cols),
                "wandb_run_id": run.id
            })
            
            print(f"  [OK] {display_name}: Avg Power = {avg_power_w:.2f} W (Across {len(power_cols)} GPU/s)")
            
        except Exception as e:
            print(f"  [ERR] {display_name}: {e}")

    # --- Save to CSV ---
    if results:
        print("\nSaving results...")
        final_df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ['model_display_name', 'avg_gpu_power_W', 'num_gpus_detected', 'raw_key', 'wandb_run_id']
        # Only select columns that actually exist in final_df
        cols = [c for c in cols if c in final_df.columns]
        
        final_df = final_df[cols]
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"SUCCESS: Data saved to {os.path.abspath(OUTPUT_FILE)}")
    else:
        print("No data extracted.")

if __name__ == "__main__":
    main()