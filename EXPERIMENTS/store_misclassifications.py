import os
import json
import torch
import gc
import pandas as pd
from collections import defaultdict
from seqeval.metrics.sequence_labeling import get_entities

# Import existing codebase functions
from EXPERIMENTS.evaluate_gold import load_and_merge_gold_data
from dataset_processing import process_llm_jsonl_results, transform_to_ner_format, CLIRENER_LABELS_V1
from span_marker import SpanMarkerModel
from gliner import GLiNER

# --- 1. CONFIGURATION ---
GOLD_DATASET_ID = "P0L3/CliReNER_v_1_1_28_GOLD"
PREDICTIONS_OUT_DIR = "RESULTS/STORED_PREDICTIONS/"
TARGET_LABELS = list(CLIRENER_LABELS_V1)

os.makedirs(PREDICTIONS_OUT_DIR, exist_ok=True)

# Map for LLM display names
LLM_NAME_MAP = {
    'gpt_5_2_pro_zs': 'GPT 5.2 Pro ZS',
    'gemini_2_5_pro_zs': 'Gemini 2.5 Pro ZS',
    'gpt_5_1_zs': 'GPT 5.1 ZS',
    'gemini_3_pro_preview_zs': 'Gemini 3.0 Pro ZS',
    'deepseek_reasoner_zs': 'DeepSeek-V3.2 (Thinking) ZS',
    'deepseek_chat_zs': 'DeepSeek-V3.2 (Non-Thinking) ZS',
    'claude_sonnet_4_5_zs': 'Claude Sonnet 4.5 ZS',
    'claude_opus_4_5_zs': 'Claude Opus 4.5 ZS'
}

# Fine-tuned models mapping directly to Hugging Face Hub IDs
# Format: (Architecture, HF_Hub_ID, Display_Name)
FINE_TUNED_MODELS = [
    ("SPANMARKER", "P0L3/CliReNER-roberta-base", "RoBERTa Base"),
    ("SPANMARKER", "P0L3/CliReNER-nasa-smd-ibm-v0.1", "INDUS Base"),
    ("SPANMARKER", "P0L3/CliReNER-indus-sde-v0.2", "INDUS SDE v0.2"),
    ("SPANMARKER", "P0L3/CliReNER-bert-base-uncased", "BERT Base"),
    ("SPANMARKER", "P0L3/CliReNER-scibert_scivocab_uncased", "SciBERT"),
    ("SPANMARKER", "P0L3/CliReNER-cliscibert_scivocab_uncased", "CliSciBERT"),
    ("SPANMARKER", "P0L3/CliReNER-clirebert_clirevocab_uncased", "CliReBERT"),
    ("SPANMARKER", "P0L3/CliReNER-distilroberta-base", "Distil RoBERTa"),
    ("SPANMARKER", "P0L3/CliReNER-EnvironmentalBERT-base", "EnvironmentalBERT"),
    ("SPANMARKER", "P0L3/CliReNER-distilroberta-base-climate-f", "ClimateBERT"),
    ("SPANMARKER", "P0L3/CliReNER-sciclimatebert", "SciClimateBERT"),
    ("GLINER", "P0L3/CliReNER-gliner_medium-v2.5", "GLiNER: Medium v2.5"),
    ("GLINER", "P0L3/CliReNER-gliner_small-v2.5", "GLiNER: Small v2.5")
]

# --- 2. HELPER FUNCTIONS ---
def extract_confusion_pairs(gold_tags, pred_tags):
    """
    Finds overlapping entities and returns a list of (Gold_Label, Pred_Label).
    Replicates the 'ent_type' evaluation strategy.
    """
    gold_ents = get_entities(gold_tags)
    pred_ents = get_entities(pred_tags)
    
    pairs = []
    matched_preds = set()
    
    for g_label, g_start, g_end in gold_ents:
        overlaps = []
        for i, (p_label, p_start, p_end) in enumerate(pred_ents):
            # Check for physical overlap
            if max(g_start, p_start) <= min(g_end, p_end):
                overlaps.append((p_label, i))
                
        if not overlaps:
            pairs.append((g_label, "O")) # Missed (False Negative)
        else:
            # If any overlapping prediction has the correct label, we count it as correct
            # Otherwise, it's a class confusion. We record the first overlapping mismatch.
            correct_overlap = any(p_lbl == g_label for p_lbl, _ in overlaps)
            if correct_overlap:
                pairs.append((g_label, g_label)) # Correct
                for p_lbl, idx in overlaps:
                    if p_lbl == g_label: matched_preds.add(idx)
            else:
                pairs.append((g_label, overlaps[0][0])) # Class Confusion
                matched_preds.add(overlaps[0][1])

    # Add Spurious (False Positives) that overlapped with nothing
    for i, (p_label, p_start, p_end) in enumerate(pred_ents):
        if i not in matched_preds:
            pairs.append(("O", p_label)) # Spurious
            
    return pairs

def run_hf_inference(model_id, model_type, texts, device):
    raw_predictions = []
    if model_type == "SPANMARKER":
        print(f"  -> Downloading/Loading SpanMarker: {model_id}")
        model = SpanMarkerModel.from_pretrained(model_id)
        if device.type == 'cuda': model.cuda()
        
        # Batch inference
        output = model.predict(texts, show_progress_bar=True)
        for i, sent_preds in enumerate(output):
            ents = [{'start': e['char_start_index'], 'end': e['char_end_index'], 'label': e['label'], 'text': e['span']} for e in sent_preds]
            raw_predictions.append({"text": texts[i], "entities": ents})
        del model
        
    elif model_type == "GLINER":
        print(f"  -> Downloading/Loading GLiNER: {model_id}")
        model = GLiNER.from_pretrained(model_id, load_tokenizer=True)
        model.to(device)
        
        for text in texts:
            out = model.predict_entities(text, TARGET_LABELS)
            raw_predictions.append({"text": text, "entities": out})
        del model
        
    torch.cuda.empty_cache()
    gc.collect()
    return raw_predictions

# --- 3. MAIN EXECUTION ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- Loading Gold Data ---")
    gold_data, bio_label_list = load_and_merge_gold_data(GOLD_DATASET_ID)
    texts = [row['text'] for row in gold_data['test']]
    
    # Track overall confusions: (Model, Gold, Pred) -> Count
    master_confusion_tally = defaultdict(int)

    # ---------------------------------------------------------
    # PART A: LLM PREDICTIONS (Zero-Shot)
    # ---------------------------------------------------------
    llm_dir = "RESULTS/LLM_PREDICTIONS/"
    if os.path.exists(llm_dir):
        for file in os.listdir(llm_dir):
            if not file.endswith(".jsonl"): continue
            
            raw_name = file.replace("ner_results_", "").replace(".jsonl", "") + "_zs"
            display_name = LLM_NAME_MAP.get(raw_name, raw_name)
            
            if "gemma_4_31b_it" in raw_name.lower(): continue # Ignore as requested
            
            out_folder = os.path.join(PREDICTIONS_OUT_DIR, display_name)
            os.makedirs(out_folder, exist_ok=True)
            out_file = os.path.join(out_folder, "hf_checkpoint_preds.json")
            
            print(f"\nProcessing LLM: {display_name}")
            
            if not os.path.exists(out_file):
                print("  -> Generating stored predictions from JSONL...")
                raw_preds = process_llm_jsonl_results(os.path.join(llm_dir, file))
                transformed_preds, _ = transform_to_ner_format(raw_preds, TARGET_LABELS)
                
                pred_lookup = {r['text']: r['ner_tags'] for r in transformed_preds if r['ner_tags']}
                
                save_data = []
                for row in gold_data['test']:
                    t_text = row['text']
                    g_tags = [bio_label_list[i] for i in row['ner_tags']]
                    p_tags = [bio_label_list[i] for i in pred_lookup.get(t_text, [0]*len(row['ner_tags']))]
                    save_data.append({"text": t_text, "tokens": row['tokens'], "gold_tags": g_tags, "pred_tags": p_tags})
                
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
            else:
                print("  -> Loading stored predictions from disk...")
                with open(out_file, 'r', encoding='utf-8') as f:
                    save_data = json.load(f)

            # Tally Confusions
            for row in save_data:
                pairs = extract_confusion_pairs(row['gold_tags'], row['pred_tags'])
                for g_lbl, p_lbl in pairs:
                    master_confusion_tally[(display_name, g_lbl, p_lbl)] += 1

    # ---------------------------------------------------------
    # PART B: SUPERVISED ENCODERS & GLINER (Via Hugging Face)
    # ---------------------------------------------------------
    for model_type, hf_id, display_name in FINE_TUNED_MODELS:
        print(f"\nProcessing Fine-Tuned Model: {display_name}")
        
        out_folder = os.path.join(PREDICTIONS_OUT_DIR, display_name)
        os.makedirs(out_folder, exist_ok=True)
        out_file = os.path.join(out_folder, "hf_checkpoint_preds.json")
        
        if not os.path.exists(out_file):
            print("  -> Running inference from Hugging Face Hub...")
            raw_preds = run_hf_inference(hf_id, model_type, texts, device)
            transformed_preds, _ = transform_to_ner_format(raw_preds, TARGET_LABELS)
            
            save_data = []
            for i, row in enumerate(gold_data['test']):
                g_tags = [bio_label_list[t] for t in row['ner_tags']]
                p_tags = [bio_label_list[t] for t in transformed_preds[i]['ner_tags']]
                save_data.append({"text": row['text'], "tokens": row['tokens'], "gold_tags": g_tags, "pred_tags": p_tags})
                
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
        else:
            print("  -> Loading stored predictions from disk...")
            with open(out_file, 'r', encoding='utf-8') as f:
                save_data = json.load(f)

        # Tally Confusions
        for row in save_data:
            pairs = extract_confusion_pairs(row['gold_tags'], row['pred_tags'])
            for g_lbl, p_lbl in pairs:
                master_confusion_tally[(display_name, g_lbl, p_lbl)] += 1

    # ---------------------------------------------------------
    # PART C: AGGREGATE AND SAVE REPORT
    # ---------------------------------------------------------
    print("\n--- Aggregating Confusion Matrix ---")
    rows = []
    for (model, g_lbl, p_lbl), count in master_confusion_tally.items():
        # We only want to export actual ERRORS (where Gold != Pred)
        if g_lbl != p_lbl and count > 0:
            rows.append({
                "Model": model,
                "Gold_Label": g_lbl,
                "Predicted_Label": p_lbl,
                "Count": count
            })
            
    df_report = pd.DataFrame(rows)
    df_report = df_report.sort_values(by=["Model", "Count"], ascending=[True, False])
    
    report_path = os.path.join(PREDICTIONS_OUT_DIR, "detailed_class_confusion_report.csv")
    df_report.to_csv(report_path, index=False)
    
    print(f"✅ Success! Stored all aligned predictions natively in JSON formats at: {PREDICTIONS_OUT_DIR}")
    print(f"✅ Confusion Matrix Tally (Errors Only) saved to: {report_path}")

if __name__ == "__main__":
    main()