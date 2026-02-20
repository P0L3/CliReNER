import pandas as pd

# --- Configuration ---
INPUT_FILE = "gpu_power_and_runtime_results.csv"
OUTPUT_FILE = "final_emissions_report.csv"

# Carbon Intensity for Croatia (gCO2eq/kWh)
# Source: https://lowcarbonpower.org/region/Croatia
CARBON_INTENSITY_CROATIA = 243.297 

def main():
    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} runs from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # --- PART A: Per-Model Aggregation ---
    # We group by model name and calculate the mean of Energy and Runtime
    model_stats = df.groupby('model_display_name').agg({
        'total_energy_kWh': 'mean',
        'runtime_seconds': 'mean',
        'avg_gpu_power_W': 'mean', # Optional: avg power across seeds
        'wandb_run_id': 'count'    # Count how many seeds exist
    }).reset_index()

    # Rename columns for clarity
    model_stats.rename(columns={
        'total_energy_kWh': 'avg_energy_per_run_kWh',
        'runtime_seconds': 'avg_runtime_seconds',
        'wandb_run_id': 'num_seeds'
    }, inplace=True)

    # Calculate CO2 per model (Average)
    # Formula: Avg kWh * Carbon Intensity
    model_stats['avg_co2_emissions_grams'] = model_stats['avg_energy_per_run_kWh'] * CARBON_INTENSITY_CROATIA

    # Sort by Energy usage (descending)
    model_stats = model_stats.sort_values(by='avg_energy_per_run_kWh', ascending=False)

    # --- PART B: Global Totals (All Experiments) ---
    # Summing the raw dataframe, not the averages
    total_energy_kwh = df['total_energy_kWh'].sum()
    total_time_seconds = df['runtime_seconds'].sum()
    
    # Time Conversions
    total_time_hours = total_time_seconds / 3600
    total_time_days = total_time_hours / 24

    # Total CO2
    total_co2_grams = total_energy_kwh * CARBON_INTENSITY_CROATIA
    total_co2_kg = total_co2_grams / 1000

    # --- PART C: Reporting ---

    print("\n" + "="*60)
    print("GLOBAL EXPERIMENT STATISTICS (All Seeds Combined)")
    print("="*60)
    print(f"1. Total Energy Spent:       {total_energy_kwh:.4f} kWh")
    print(f"2. Total CO2 Emissions:      {total_co2_grams:.2f} g ({total_co2_kg:.4f} kg)")
    print(f"3. Total Duration:           {total_time_seconds:.0f} seconds")
    print(f"                             = {total_time_hours:.2f} hours")
    print(f"                             = {total_time_days:.2f} days")
    print(f"4. Total Runs Processed:     {len(df)}")
    print("="*60 + "\n")

    print("PER-MODEL AVERAGES (Sorted by Energy Impact):")
    print("-" * 110)
    print(f"{'Model Name':<30} | {'Avg kWh':<12} | {'Avg CO2 (g)':<12} | {'Avg Time (s)':<12} | {'Avg Power (W)':<12}")
    print("-" * 110)

    for _, row in model_stats.iterrows():
        print(f"{row['model_display_name']:<30} | "
              f"{row['avg_energy_per_run_kWh']:<12.6f} | "
              f"{row['avg_co2_emissions_grams']:<12.2f} | "
              f"{row['avg_runtime_seconds']:<12.0f} | "
              f"{row['avg_gpu_power_W']:<12.2f}")
    print("-" * 110)

    # Save detailed per-model stats to CSV
    model_stats.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDetailed per-model stats saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()