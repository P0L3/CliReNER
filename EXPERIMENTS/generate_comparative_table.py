import argparse
import torch
import random
import sys
import gc
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset, concatenate_datasets
from span_marker import SpanMarkerModel
from gliner import GLiNER

# Import path resolution logic
from EXPERIMENTS.finetune import get_output_dir
from dataset_processing import shorten_name

# --- CONFIGURATION ---
DEFAULT_SEEDS = [0, 42, 3012, 33, 131]

all_tags = """Asset
Body of Water
Body Part
Chemical
Disease
Ecosystem
Energy Source
Field of Study
Geographical Feature
Intellectual Artefact
Location
Mathematical Expression
Measuring Device
Meteorological Phenomenon
Method
Natural Disaster
Natural Phenomenon
Organism
Organization
Other
Person
Physical Artefact
Physical Phenomenon
Policy or Objective
Quantity
Satellite
System
Time Period"""

TARGET_TAGS = all_tags.split("\n")

# --- THRESHOLDS (The 4-1 Rule) ---
SUCCESS_THRESHOLD = 4  # Must be found in >= 4 seeds
FAILURE_THRESHOLD = 1  # Must be found in <= 1 seed

def load_gold_data(dataset_id):
    print(f"--- Loading Dataset: {dataset_id} ---")
    ds = load_dataset(dataset_id)
    splits = [ds[split] for split in ds.keys()]
    merged = concatenate_datasets(splits)
    features = merged.features["ner_tags"].feature.names
    id2label = {i: name for i, name in enumerate(features)}
    return merged, id2label

def extract_gold_spans(row, id2label):
    tokens = row['tokens']
    tags = row['ner_tags']
    extracted_gold = []
    curr_start, curr_label = None, None
    for i, tag_id in enumerate(tags):
        tag_name = id2label[tag_id]
        if tag_name.startswith("B-"):
            if curr_label: extracted_gold.append((" ".join(tokens[curr_start:i]), curr_label))
            curr_start, curr_label = i, tag_name[2:]
        elif tag_name.startswith("I-") and curr_label == tag_name[2:]: continue
        else:
            if curr_label: extracted_gold.append((" ".join(tokens[curr_start:i]), curr_label))
            curr_label, curr_start = None, None
    if curr_label: extracted_gold.append((" ".join(tokens[curr_start:]), curr_label))
    return set(extracted_gold)

def run_inference_single_seed(model_path, model_type, texts, device):
    predictions = []
    if model_type == "SPANMARKER":
        model = SpanMarkerModel.from_pretrained(model_path)
        if device.type == 'cuda': model.cuda()
        output = model.predict(texts, show_progress_bar=False)
        for sent_preds in output:
            predictions.append({(ent['span'], ent['label']) for ent in sent_preds})
        del model
    elif model_type == "GLINER":
        model = GLiNER.from_pretrained(model_path)
        model.to(device)
        from dataset_processing import CLIRENER_LABELS_V1
        labels = list(CLIRENER_LABELS_V1)
        for text in texts:
            out = model.predict_entities(text, labels, threshold=0.5)
            predictions.append({(ent['text'], ent['label']) for ent in out})
        del model
    torch.cuda.empty_cache()
    return predictions

def get_seed_frequencies(base_dir, model_type, model_id, train_dataset, texts, device, seeds):
    print(f"\n>>> Analyzing Model: {model_id}")
    freq_maps = [Counter() for _ in texts]
    for seed in seeds:
        path = get_output_dir(base_dir, model_type, model_id, train_dataset, seed) / "checkpoint-final"
        if not path.exists(): continue
        print(f"  - Processing Seed {seed}...")
        seed_preds = run_inference_single_seed(str(path), model_type, texts, device)
        for idx, sent_set in enumerate(seed_preds):
            freq_maps[idx].update(sent_set)
        gc.collect()
    return freq_maps

def format_context(text, entity_text):
    start = text.find(entity_text)
    if start == -1: return f"**{entity_text}**"
    end = start + len(entity_text)
    ctx_start, ctx_end = max(0, start - 35), min(len(text), end + 35)
    snippet = text[ctx_start:ctx_end]
    formatted = snippet.replace(entity_text, f"**{entity_text}**")
    return f"{'...' if ctx_start > 0 else ''}{formatted}{'...' if ctx_end < len(text) else ''}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_id", type=str, required=True)
    parser.add_argument("--challenger_id", type=str, required=True)
    parser.add_argument("--baseline_type", type=str, default="SPANMARKER")
    parser.add_argument("--challenger_type", type=str, default="SPANMARKER")
    parser.add_argument("--eval_dataset", type=str, default="P0L3/CliReNER_v_1_1_28_GOLD_authorannots")
    parser.add_argument("--train_dataset", type=str, default="P0L3/CliReNER_v_1_1_28_SILVER")
    parser.add_argument("--output_dir", type=str, default="RESULTS/QUALITATIVE_MODEL_COMPARISON", help="Directory to save the resulting .md file")
    args = parser.parse_args()

    # --- Construct Naming Convention ---
    base_short = shorten_name(args.baseline_id)
    chall_short = shorten_name(args.challenger_id)
    # Format: base_vs_chall_S4_F1.md
    filename = f"{base_short}_vs_{chall_short}_S{SUCCESS_THRESHOLD}_F{FAILURE_THRESHOLD}.md"
    output_path = Path(args.output_dir) / filename

    # 1. Load Data
    dataset, id2label = load_gold_data(args.eval_dataset)
    texts, gold_sets = dataset['text'], [extract_gold_spans(row, id2label) for row in dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Get frequencies
    base_freqs = get_seed_frequencies("EXPERIMENTS/models", args.baseline_type, args.baseline_id, args.train_dataset, texts, device, DEFAULT_SEEDS)
    chall_freqs = get_seed_frequencies("EXPERIMENTS/models", args.challenger_type, args.challenger_id, args.train_dataset, texts, device, DEFAULT_SEEDS)

    # 3. Categorization
    results = {tag: {"lost": [], "shared": [], "gained": []} for tag in TARGET_TAGS}

    print(f"\n--- Categorizing (4-1 Rule: Success >={SUCCESS_THRESHOLD}, Failure <={FAILURE_THRESHOLD}) ---")
    
    for i, text in enumerate(texts):
        G = gold_sets[i]
        B_map, C_map = base_freqs[i], chall_freqs[i]

        for (ent_text, ent_label) in G:
            if ent_label not in TARGET_TAGS: continue
            
            b_votes = B_map.get((ent_text, ent_label), 0)
            c_votes = C_map.get((ent_text, ent_label), 0)
            
            ctx = format_context(text, ent_text)

            if b_votes >= SUCCESS_THRESHOLD and c_votes >= SUCCESS_THRESHOLD:
                results[ent_label]["shared"].append(ctx)
            elif b_votes >= SUCCESS_THRESHOLD and c_votes <= FAILURE_THRESHOLD:
                results[ent_label]["lost"].append(ctx)
            elif c_votes >= SUCCESS_THRESHOLD and b_votes <= FAILURE_THRESHOLD:
                results[ent_label]["gained"].append(ctx)

    # 4. Generate Table
    print(f"--- Writing to {output_path} ---")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Robust Architectural Comparison: {args.baseline_id} vs {args.challenger_id}\n")
        f.write(f"- **Strictness (4-1 Rule):**\n")
        f.write(f"  - **Stable Success:** Found in $\ge {SUCCESS_THRESHOLD}/5$ seeds.\n")
        f.write(f"  - **Stable Failure:** Found in $\le {FAILURE_THRESHOLD}/5$ seeds.\n\n")
        
        f.write(f"| Entity Type | Correct in Baseline ONLY ({base_short}) | Correct in BOTH | Correct in Challenger ONLY ({chall_short}) |\n")
        f.write("|---|---|---|---|\n")
        
        for tag in TARGET_TAGS:
            def sample_cell(key):
                items = list(set(results[tag][key]))
                if not items: return "*(None found matching 4-1 rule)*"
                return "<br><br>".join(random.sample(items, min(2, len(items))))
            
            f.write(f"| **{tag}** | {sample_cell('lost')} | {sample_cell('shared')} | {sample_cell('gained')} |\n")

    print("Done.")

if __name__ == "__main__":
    main()