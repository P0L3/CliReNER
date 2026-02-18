import subprocess
import sys
from itertools import combinations

# 1. ID Mapping
ID_MAP = {
    "RoBERTa Base": "FacebookAI/roberta-base",
    "INDUS Base": "nasa-impact/nasa-smd-ibm-v0.1",
    "INDUS SDE v0.2": "nasa-impact/indus-sde-v0.2",
    
    "BERT Base": "google-bert/bert-base-uncased",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "CliSciBERT": "P0L3/cliscibert_scivocab_uncased",
    "CliReBERT": "P0L3/clirebert_clirevocab_uncased",
    
    "Distil RoBERTa": "distilbert/distilroberta-base",
    "EnvironmentalBERT": "ESGBERT/EnvironmentalBERT-base",
    "ClimateBERT": "climatebert/distilroberta-base-climate-f",
    "SciClimateBERT": "P0L3/sciclimatebert"
}

# 2. Families defined as lists of display names
FAMILIES = [
    ["RoBERTa Base", "INDUS Base", "INDUS SDE v0.2"],
    ["BERT Base", "SciBERT", "CliSciBERT", "CliReBERT"],
    ["Distil RoBERTa", "EnvironmentalBERT", "ClimateBERT", "SciClimateBERT"]
]

def main():
    for group in FAMILIES:
        # Generate every unique pair within the group
        for model_a_name, model_b_name in combinations(group, 2):
            
            print(f"\n>>> Comparing: {model_a_name} vs {model_b_name}")
            
            cmd = [
                sys.executable, "-m", "EXPERIMENTS.generate_comparative_table",
                "--baseline_id", ID_MAP[model_a_name],
                "--challenger_id", ID_MAP[model_b_name],
                "--baseline_type", "SPANMARKER",
                "--challenger_type", "SPANMARKER"
            ]
            
            subprocess.run(cmd)

if __name__ == "__main__":
    main()