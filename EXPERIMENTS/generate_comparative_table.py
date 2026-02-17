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

# Import path resolution logic from your framework
from EXPERIMENTS.finetune import get_output_dir

# --- CONFIGURATION ---
DEFAULT_SEEDS = [0, 42, 3012, 33, 131]
TARGET_TAGS = [
    "Body Part", "Organization", "Ecosystem",  # High Loss types
    "Method", "Intellectual Artefact"          # High Gain types
]

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
    """Returns a list of Counters: maps (entity, label) to number of seeds that found it."""
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
    parser.add_argument("--output", type=str, default="PLOTS/hardened_comparison.md")
    args = parser.parse_args()

    dataset, id2label = load_gold_data(args.eval_dataset)
    texts, gold_sets = dataset['text'], [extract_gold_spans(row, id2label) for row in dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get frequencies (0 to 5) for every potential entity
    base_freqs = get_seed_frequencies("EXPERIMENTS/models", args.baseline_type, args.baseline_id, args.train_dataset, texts, device, DEFAULT_SEEDS)
    chall_freqs = get_seed_frequencies("EXPERIMENTS/models", args.challenger_type, args.challenger_id, args.train_dataset, texts, device, DEFAULT_SEEDS)

    results = {tag: {"lost": [], "shared": [], "gained": []} for tag in TARGET_TAGS}

    for i, text in enumerate(texts):
        G = gold_sets[i]
        B_map, C_map = base_freqs[i], chall_freqs[i]

        for (ent_text, ent_label) in G:
            if ent_label not in TARGET_TAGS: continue
            
            b_votes = B_map.get((ent_text, ent_label), 0)
            c_votes = C_map.get((ent_text, ent_label), 0)
            
            ctx = format_context(text, ent_text)

            # HARDENED LOGIC:
            # Shared: Both had Majority (>=3)
            if b_votes >= 3 and c_votes >= 3:
                results[ent_label]["shared"].append(ctx)
            # Lost: Baseline had Majority (>=3) AND Challenger was completely blind (0)
            elif b_votes >= 3 and c_votes == 0:
                results[ent_label]["lost"].append(ctx)
            # Gained: Challenger had Majority (>=3) AND Baseline was completely blind (0)
            elif c_votes >= 3 and b_votes == 0:
                results[ent_label]["gained"].append(ctx)

    print(f"--- Writing to {args.output} ---")
    with open(args.output, "w") as f:
        f.write(f"# Hardened Architectural Comparison: {args.baseline_id} vs {args.challenger_id}\n")
        f.write("- **Criteria for 'Correct':** Found in $\ge 3$ out of 5 seeds.\n")
        f.write("- **Criteria for 'Blind':** Found in 0 out of 5 seeds.\n\n")
        f.write("| Entity Type | Baseline Correct & Challenger Blind | Shared (Both Correct) | Challenger Correct & Baseline Blind |\n")
        f.write("|---|---|---|---|\n")
        for tag in TARGET_TAGS:
            def sample_cell(key):
                items = list(set(results[tag][key])) # Unique strings
                if not items: return "*(None Found)*"
                return "<br><br>".join(random.sample(items, min(2, len(items))))
            f.write(f"| **{tag}** | {sample_cell('lost')} | {sample_cell('shared')} | {sample_cell('gained')} |\n")

if __name__ == "__main__":
    main()