import wandb
import pandas as pd
import re
import os

# 1. Configuration
ENTITY = "andrija-2"
PROJECT = "CLIRENER_SILVER_SEEDS"
OUTPUT_FILE = "gpu_power_and_runtime_results.csv"

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
    
    print(f"Found {len(runs)} runs. Processing Power & Runtime...")

    for run in runs:
        raw_key = clean_model_name(run.name)
        display_name = MODEL_NAME_MAP.get(raw_key, raw_key)
        
        # --- 1. Fetch Runtime (Duration) ---
        # Try getting it from the summary first (fastest)
        runtime_s = run.summary.get("_runtime")
        
        # If not in summary, we'll try to find it in the history later
        
        try:
            # --- 2. Fetch GPU Metrics ---
            try:
                # Fetching 'events' (system metrics)
                events_df = run.history(stream="events", samples=100000)
            except Exception:
                events_df = pd.DataFrame()
            
            if events_df.empty:
                # Fallback to default history
                events_df = run.history(samples=100000)
                
            if events_df.empty:
                print(f"  [SKIP] {display_name}: No history data found.")
                continue

            # --- Runtime Fallback ---
            # If summary didn't have runtime, get the max '_runtime' from the dataframe
            if runtime_s is None and "_runtime" in events_df.columns:
                runtime_s = events_df["_runtime"].max()
            
            # If we still don't have runtime, we can't calculate energy
            if runtime_s is None:
                print(f"  [WARN] {display_name}: Could not determine runtime.")
                runtime_s = 0

            # --- 3. Process GPU Power ---
            # Identify columns containing 'powerWatts'
            power_cols = [c for c in events_df.columns if "powerWatts" in c]
            
            avg_power_w = 0.0
            num_gpus = 0
            
            if power_cols:
                # Filter to ONLY power columns and drop rows where they are all NaN
                power_data = events_df[power_cols].dropna(how='all')
                
                if not power_data.empty:
                    # Sum across GPUs (axis=1) then Average over time (mean)
                    total_power_per_step = power_data.sum(axis=1)
                    avg_power_w = total_power_per_step.mean()
                    num_gpus = len(power_cols)
            else:
                # This is expected for API models or CPU-only runs
                # print(f"  [INFO] {display_name}: No GPU power metrics (likely API/CPU).")
                pass

            # --- 4. Calculate Energy ---
            # Energy (Joules) = Power (Watts) * Time (Seconds)
            total_energy_joules = avg_power_w * runtime_s
            
            # Energy (kWh) = Joules / 3,600,000
            total_energy_kwh = total_energy_joules / 3600000

            results.append({
                "model_display_name": display_name,
                "raw_key": raw_key,
                "runtime_seconds": round(runtime_s, 2),
                "avg_gpu_power_W": round(avg_power_w, 2),
                "total_energy_kWh": total_energy_kwh, # High precision useful here
                "num_gpus_detected": num_gpus,
                "wandb_run_id": run.id
            })
            
            if num_gpus > 0:
                print(f"  [OK] {display_name}: {runtime_s:.0f}s | {avg_power_w:.1f}W | {total_energy_kwh:.6f} kWh")
            else:
                print(f"  [OK] {display_name}: {runtime_s:.0f}s | (No GPU)")

        except Exception as e:
            print(f"  [ERR] {display_name}: {e}")

    # --- Save to CSV ---
    if results:
        print("\nSaving results...")
        final_df = pd.DataFrame(results)
        
        # Order columns logically
        desired_order = [
            'model_display_name', 
            'runtime_seconds', 
            'avg_gpu_power_W', 
            'total_energy_kWh', 
            'num_gpus_detected',
            'raw_key',
            'wandb_run_id'
        ]
        
        # Select columns that exist
        final_cols = [c for c in desired_order if c in final_df.columns]
        final_df = final_df[final_cols]
        
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"SUCCESS: Data saved to {os.path.abspath(OUTPUT_FILE)}")
    else:
        print("No data extracted.")

if __name__ == "__main__":
    main()