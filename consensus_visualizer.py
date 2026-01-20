import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import os
from os import listdir
from pathlib import Path
from collections import defaultdict, Counter

# ==============================================================================
# 1. GLOBAL CONFIGURATION & CONSTANTS
# ==============================================================================

st.set_page_config(layout="wide", page_title="Consensus Tuner")

# Paths
ANNOTATOR_DIR = "/home/p0l3/RAD/DROP/CLIRENER/ANNOTATORS/"
CLIRENER_DIR = "" 

# Default Importance for processing
CLIRENER_ANNOTATOR_IMPORTANCE = [2, 1, 3]

# Label Definitions
CLIRENER_LABELS_V1 = {
    "Ecosystem", "Energy Source", "Natural Disaster", "Meteorological Phenomenon", 
    "Quantity", "Intellectual Artefact", "Body of Water", "Disease", "Location", 
    "Physical Phenomenon", "Chemical", "Time Period", "Organization", 
    "Natural Phenomenon", "Field of Study", "Mathematical Expression", 
    "Measuring Device", "Geographical Feature", "System", "Satellite", "Organism", 
    "Method", "Other", "Person", "Physical Artefact", "Body Part", "Asset", "Policy", 
}

# UI Colors
LABEL_COLORS = {
    "Asset": "#B06C23", "Body Part": "#6ebe9e", "Body of Water": "#068d96",
    "Chemical": "#8426bf", "Disease": "#f7300a", "Ecosystem": "#abebc6",
    "Energy Source": "#f7ef0a", "Field of Study": "#262626", "Geographical Feature": "#b36004",
    "Intellectual Artefact": "#c903AB", "Location": "#afa79e", "Mathematical Expression": "#ad26bf",
    "Measuring Device": "#f7c80a", "Meteorological Phenomenon": "#0ae7f7", "Method": "#F759AB",
    "Natural Disaster": "#f7750a", "Natural Phenomenon": "#38bf26", "Organism": "#6eba6a",
    "Organization": "#b09223", "Other": "#e9dff2", "Person": "#b59297",
    "Physical Artefact": "#B593AC", "Physical Phenomenon": "#ba26bf", "Policy": "#B0A223",
    "Quantity": "#949494", "Satellite": "#101247", "System": "#171200", "Time Period": "#4d4d4d"
}

# ==============================================================================
# 2. CORE PROCESSING FUNCTIONS (FRAGILE - UNTOUCHED LOGIC)
# ==============================================================================

def tokenize_text(text):
    """Tokenizes the input text into a list of tokens using a specific regex."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

def cwed4eta_process_json_file(file_path=CLIRENER_DIR, annotator_importance=CLIRENER_ANNOTATOR_IMPORTANCE): 
    with open(file_path) as f:
        json_string = json.load(f)
    
    dataset = []
    for task in json_string:
        
        if type(annotator_importance) != type(CLIRENER_ANNOTATOR_IMPORTANCE):
            predictions = task.get('predictions')
            selected_annotations = [p.get('result') for p in predictions][0]
        else:
            if not task.get('annotations'):
                continue
            selected_annotations = None
            annotations_by_id = {an['completed_by']: an for an in task.get('annotations', [])}
            
            for annotator_id in annotator_importance:
                if annotator_id in annotations_by_id:
                    selected_annotations = annotations_by_id[annotator_id].get('result', [])
                    break

            if selected_annotations is None:
                continue
        
        text = task["data"]["sentence"]
        entities = []
        for annotation in selected_annotations:
            if annotation.get('type') != 'labels':
                continue
            value = annotation.get('value', {})
            if 'text' in value and 'labels' in value and value['labels']:
                entities.append({
                    'text': value["text"],
                    'label': value["labels"][0],
                    'start': value["start"],
                    'end': value["end"]
                })
        paper_id = task["data"]["paper_id"]
        sentence_id = task["data"]["sentence_id"]
        
        compound_id = f"{paper_id}-{sentence_id}"
        dataset.append({
            "text": text,
            "entities": entities,
            "id": compound_id
        })
        
    return dataset

def convert_to_token_spans(structured_docs_with_char_spans):
    final_data = []

    for doc in structured_docs_with_char_spans:
        text = doc['text']
        id_val = doc.get("id")
        
        new_tokens = tokenize_text(text)
        
        new_token_char_spans = []
        current_offset = 0
        for token in new_tokens:
            start = text.find(token, current_offset)
            end = start + len(token)
            new_token_char_spans.append((start, end))
            current_offset = end

        doc_entities_token_spans = []
        for entity in doc['entities']:
            entity_char_start = entity['start']
            entity_char_end = entity['end']
            
            aligned_token_indices = []
            for i, (token_char_start, token_char_end) in enumerate(new_token_char_spans):
                if max(entity_char_start, token_char_start) < min(entity_char_end, token_char_end):
                    aligned_token_indices.append(i)
            
            if aligned_token_indices:
                token_span_start = aligned_token_indices[0]
                token_span_end = aligned_token_indices[-1]
                doc_entities_token_spans.append([token_span_start, token_span_end, entity["label"]])
        
        entry = {
            "text": text,
            "tokenized_text": new_tokens,
            "ner": doc_entities_token_spans
        }
        if id_val:
            entry["id"] = id_val
            
        final_data.append(entry)
        
    return final_data

def process_directory_of_json_files(directory_path, annotator_importance_list):
    # Ensure standard path format
    if not directory_path.endswith("/"): 
        directory_path += "/"
        
    json_files = [file for file in listdir(directory_path) if file.endswith("json")]
    compound_json = []
    
    for file in json_files:
        compound_json.extend(cwed4eta_process_json_file(str(Path(directory_path + file)), annotator_importance_list))
        
    return compound_json

# ==============================================================================
# 3. CONSENSUS LOGIC HELPERS
# ==============================================================================

def spans_to_bio_tags(token_list, ner_spans):
    tags = ["O"] * len(token_list)
    if not ner_spans:
        return tags
        
    # Check if span is 3-element [start, end, label] or 4 [start, end, label, tie]
    # NOTE: Input here usually assumes Inclusive End from convert_to_token_spans
    
    for item in ner_spans:
        start = item[0]
        end = item[1]
        label = item[2]
            
        b_tag = f"B-{label}"
        i_tag = f"I-{label}"
        
        tags[start] = b_tag
        if end > start:
            for i in range(start + 1, end + 1):
                tags[i] = i_tag
    return tags

def bio_tags_to_spans_with_ties(tags, token_ties):
    """ 
    Converts BIO tags and a parallel list of booleans (is_tie) into spans.
    Returns: [[start_token_idx, end_token_idx, label, is_tie_boolean], ...]
    """
    spans = []
    current_start = None
    current_label = None
    
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_label is not None:
                # Check if any token in the previous span was a tie
                span_tie = any(token_ties[current_start : i])
                spans.append([current_start, i - 1, current_label, span_tie])
            
            current_start = i
            current_label = tag[2:]
            
        elif tag.startswith("I-"):
            new_label = tag[2:]
            if current_label is None:
                current_start = i
                current_label = new_label
            elif current_label != new_label:
                # Label switch
                span_tie = any(token_ties[current_start : i])
                spans.append([current_start, i - 1, current_label, span_tie])
                current_start = i
                current_label = new_label
                
        else: # "O"
            if current_label is not None:
                span_tie = any(token_ties[current_start : i])
                spans.append([current_start, i - 1, current_label, span_tie])
                current_label = None
                current_start = None

    if current_label is not None:
        span_tie = any(token_ties[current_start : len(tags)])
        spans.append([current_start, len(tags) - 1, current_label, span_tie])
        
    return spans

def get_expert_mapping():
    groups = {
        "G1": {"annotators": [5, 6], "labels": {"Asset", "Policy", "Objective", "Method", "Field of Study", "Intellectual Artefact"}},
        "G2": {"annotators": [4, 9], "labels": {"Location", "Geographical Feature", "Body of Water", "Time Period", "Satellite"}},
        "G3": {"annotators": [1, 10], "labels": {"Mathematical Expression", "Measuring Device", "Physical Phenomenon", "Quantity"}},
        "G4": {"annotators": [2, 3], "labels": {"Body Part", "Chemical", "Disease", "Organism", "Ecosystem"}},
        "G5": {"annotators": [11], "labels": {"Energy Source", "Meteorological Phenomenon", "Natural Disaster", "Natural Phenomenon"}},
        "G6": {"annotators": [7, 5, 8], "labels": {"Physical Artefact", "Organization", "Person", "System"}}
    }
    expert_map = defaultdict(set)
    for g_name, data in groups.items():
        for annotator_id in data["annotators"]:
            expert_map[annotator_id].update(data["labels"])
    return expert_map

def generate_consensus_dataset(list_of_annotator_data, weights=(1.0, 0.7), o_weight=None):
    expert_weight, non_expert_weight = weights
    final_o_weight = o_weight if o_weight is not None else non_expert_weight
    expert_map = get_expert_mapping()
    
    aligned_docs = defaultdict(dict)
    for ann_idx, dataset in enumerate(list_of_annotator_data):
        if dataset is None: continue 
        for doc in dataset:
            aligned_docs[doc['id']][ann_idx] = doc

    consensus_dataset = []

    for doc_id, ann_data_map in aligned_docs.items():
        ref_doc = list(ann_data_map.values())[0]
        tokens = ref_doc['tokenized_text']
        doc_len = len(tokens)

        annotator_tags = {}
        for ann_idx, doc in ann_data_map.items():
            if len(doc['tokenized_text']) == doc_len:
                annotator_tags[ann_idx] = spans_to_bio_tags(tokens, doc['ner'])
        
        num_raters = len(annotator_tags)
        final_tags = []
        token_ties = []
        token_scores = []

        # =========================================================
        # 1. Vote per token (Two-Stage Voting)
        # =========================================================
        for t_i in range(doc_len):
            bio_scores = defaultdict(float)      
            semantic_scores = defaultdict(float) 

            for ann_idx, tags_list in annotator_tags.items():
                tag = tags_list[t_i]
                if tag == "O":
                    w = final_o_weight
                    label_core = "O"
                else:
                    label_core = tag[2:] 
                    if label_core in expert_map.get(ann_idx, set()):
                        w = expert_weight
                    else:
                        w = non_expert_weight
                
                bio_scores[tag] += w
                semantic_scores[label_core] += w
            
            token_scores.append(dict(bio_scores))

            if not semantic_scores:
                final_tags.append("O")
                token_ties.append(False)
                continue

            candidates = list(semantic_scores.items())
            candidates.sort(key=lambda x: (-x[1], x[0] == "O", x[0]))
            
            winning_category = candidates[0][0]
            
            is_tie = False
            if len(candidates) > 1 and candidates[0][1] == candidates[1][1]:
                is_tie = True
            token_ties.append(is_tie)

            if winning_category == "O":
                final_tags.append("O")
            else:
                b_tag = f"B-{winning_category}"
                i_tag = f"I-{winning_category}"
                # Winner takes all for boundary
                if bio_scores.get(b_tag, 0.0) >= bio_scores.get(i_tag, 0.0):
                    final_tags.append(b_tag)
                else:
                    final_tags.append(i_tag)

        # =========================================================
        # 2. Post-process tags (Consistency)
        # =========================================================
        cleaned_tags = []
        for i, tag in enumerate(final_tags):
            if tag.startswith("I-"):
                label = tag[2:]
                prev = cleaned_tags[i-1] if i > 0 else "O"
                if prev not in [f"B-{label}", f"I-{label}"]:
                    cleaned_tags.append(f"B-{label}")
                else:
                    cleaned_tags.append(tag)
            else:
                cleaned_tags.append(tag)

        # =========================================================
        # 3. Create Spans
        # =========================================================
        consensus_spans = bio_tags_to_spans_with_ties(cleaned_tags, token_ties)
        
        # 4. ENRICH SPANS WITH VOTE STATS
        # Adjusted for List Format: [start, end_inclusive, label, tie, stats]
        for span in consensus_spans:
            s = span[0]
            e_inclusive = span[1] # Inclusive
            
            span_vote_sum = defaultdict(float)
            
            # Use range(s, e + 1) because e is inclusive
            for t_idx in range(s, e_inclusive + 1):
                t_scores = token_scores[t_idx]
                for tag, val in t_scores.items():
                    clean_lbl = tag[2:] if tag != "O" else "O"
                    span_vote_sum[clean_lbl] += val
            
            # Normalize
            current_span_len = max(1, (e_inclusive - s) + 1)
            stats = {k: round(v/current_span_len, 2) for k, v in span_vote_sum.items()}
            
            # Append stats as the 5th element
            span.append(stats)

        consensus_dataset.append({
            "id": doc_id,
            "text": ref_doc.get("text", ""),
            "tokenized_text": tokens,
            "ner": consensus_spans,
            "num_raters": num_raters,
            "scores": token_scores
        })

    return consensus_dataset

# ==============================================================================
# 4. DATA LOADING 
# ==============================================================================

@st.cache_data
def load_data_cached():
    """Loads data using a configuration map to ensure cleaner structure."""
    
    st.write(f"📂 Working with root: `{ANNOTATOR_DIR}`")

    # Configuration for data loading
    # Format: Index: {path, ids, is_dir}
    DATA_CONFIG = {
        0: None, # Explicitly requested as None
        1:  {"path": "1/G3_19126.json",   "ids": [1, 5]},
        2:  {"path": "2/G4_14126.json",   "ids": [1, 6]},
        3:  {"path": "3/G4_19126.json",   "ids": [1, 7]},
        4:  {"path": "4/G2_7126.json",    "ids": [8]},
        5:  {"path": "5/",                "ids": [9], "is_dir": True},
        6:  {"path": "6/G1_15126.json",   "ids": [13]},
        7:  None,
        8:  {"path": "8/G6_7126.json",    "ids": [12]},
        9:  None,
        10: {"path": "10/G3_19126.json",  "ids": [1, 15]},
        11: {"path": "11/G5_19126.json",  "ids": [4, 1, 14]},
        12: None
    }

    def safe_load(subpath, annotator_ids, is_directory=False):
        full_path = os.path.join(ANNOTATOR_DIR, subpath)
        if not os.path.exists(full_path):
            st.error(f"❌ Path not found: {full_path}")
            return None
        
        try:
            if is_directory:
                raw_data = process_directory_of_json_files(full_path, annotator_ids)
            else:
                raw_data = cwed4eta_process_json_file(full_path, annotator_ids)
            return convert_to_token_spans(raw_data)
        except Exception as e:
            st.error(f"⚠️ Error loading {subpath}: {e}")
            return None

    all_data = []
    
    # Iterate 0 to 12 to maintain consistent list indices
    for i in range(13):
        config = DATA_CONFIG.get(i)
        
        if config is None:
            all_data.append(None)
        else:
            loaded_data = safe_load(
                config["path"], 
                config["ids"], 
                is_directory=config.get("is_dir", False)
            )
            all_data.append(loaded_data)
            
    return all_data

# ==============================================================================
# 5. UI COMPONENTS (HTML RENDERER)
# ==============================================================================

def render_ner_html(text, tokens, spans):
    """
    Renders the text with colored highlights.
    Adapts to List-based spans [start, end_inclusive, label, tie, stats]
    """
    token_styles = [None] * len(tokens)
    
    for span in spans:
        start_t = span[0]
        end_t_inclusive = span[1] # Inclusive
        label = span[2]
        is_tie = span[3]
        
        color = LABEL_COLORS.get(label, "#ccc")
        border_style = "2px dashed red" if is_tie else "none"
        
        # Iterate including end_t_inclusive
        for i in range(start_t, end_t_inclusive + 1):
            position = "middle"
            if i == start_t: position = "start"
            if i == end_t_inclusive: position = "end"
            if start_t == end_t_inclusive: position = "single"
            
            token_styles[i] = {
                "color": color, 
                "label": label if position in ["end", "single"] else None,
                "position": position,
                "border": border_style
            }

    html = '<div style="line-height: 2.5; font-family: sans-serif;">'
    for i, token in enumerate(tokens):
        style = token_styles[i]
        if style:
            border_radius = "0"
            padding = "2px 0px"
            if style["position"] == "start": 
                border_radius = "4px 0 0 4px"
                padding = "2px 0px 2px 6px"
            if style["position"] == "end": 
                border_radius = "0 4px 4px 0"
                padding = "2px 6px 2px 0px"
            if style["position"] == "single": 
                border_radius = "4px"
                padding = "2px 6px"

            text_color = "white" if style["color"] not in ["#e9dff2", "#abebc6", "#f7ef0a"] else "black"

            html += f'<span style="background-color: {style["color"]}; color: {text_color}; padding: {padding}; border-radius: {border_radius}; margin: 0 1px; border-bottom: {style["border"]}">{token}'
            if style["label"]:
                html += f'<span style="font-size: 0.6em; font-weight: bold; margin-left: 5px; text-transform: uppercase; opacity: 0.9;">{style["label"]}</span>'
            html += '</span>'
        else:
            html += f'<span>{token}</span>'
        html += " "
    html += '</div>'
    return html

# ==============================================================================
# 6. MAIN APPLICATION EXECUTION
# ==============================================================================

def main():
    st.title("⚖️ Live Consensus Tuner")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Weights")
        exp_w = st.slider("Expert Weight", 0.0, 5.0, 1.0, 0.1)
        non_exp_w = st.slider("Non-Expert Weight", 0.0, 5.0, 0.7, 0.1)
        
        use_custom_o = st.checkbox("Custom 'O' Weight?", value=False)
        if use_custom_o:
            o_w = st.slider("No-Entity (O) Weight", 0.0, 5.0, 0.9, 0.1)
            final_o_w = o_w
        else:
            final_o_w = None
            st.info(f"O-Weight: {non_exp_w} (Default)")

        st.divider()
        st.markdown("### Legend")
        cols = st.columns(2)
        for i, (label, color) in enumerate(LABEL_COLORS.items()):
            cols[i % 2].markdown(f"<span style='color:{color}'>■</span> {label}", unsafe_allow_html=True)

    # --- Load Data ---
    raw_data_list = load_data_cached()

    # --- Run Consensus ---
    consensus_data = generate_consensus_dataset(
        raw_data_list, 
        weights=(exp_w, non_exp_w), 
        o_weight=final_o_w
    )

    # --- Display Logic ---
    doc_map = {d['id']: d for d in consensus_data}
    if not doc_map:
        st.error("No documents loaded. Check file paths and JSON structure.")
        st.stop()

    st.subheader("Visual Inspection")
    selected_doc_id = st.selectbox("Select Document", list(doc_map.keys()))

    if selected_doc_id:
        doc = doc_map[selected_doc_id]
        
        # Metadata
        st.markdown(f"**Document ID:** `{selected_doc_id}` | **Num Raters:** {doc['num_raters']}")
        
        # HTML Render
        html_view = render_ner_html(doc['text'], doc['tokenized_text'], doc['ner'])
        st.markdown(html_view, unsafe_allow_html=True)
        
        st.divider()
        
        # --- Entity Report Table ---
        st.subheader("📊 Detailed Entity Vote Report")
        
        if doc['ner']:
            # Construct DataFrame for friendly display
            report_data = []
            tokens = doc['tokenized_text']
            
            for span in doc['ner']:
                # Unpack List: [start, end, label, tie, stats]
                start = span[0]
                end = span[1] # Inclusive
                label = span[2]
                tie = span[3]
                stats = span[4] if len(span) > 4 else {}

                # Join tokens (Slice is exclusive, so use end+1)
                entity_text = " ".join(tokens[start : end + 1])
                
                # Format the stats string
                stats_str = ", ".join([f"{k}: {v}" for k,v in stats.items()])
                
                report_data.append({
                    "Entity Text": entity_text,
                    "Consensus Label": label,
                    "Tie?": "YES" if tie else "No",
                    "Vote Breakdown (Avg Weight)": stats_str
                })
            
            df = pd.DataFrame(report_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No entities found in this document based on current weights.")

        # --- Debugging ---
        with st.expander("Raw Debug Info"):
            st.write("Calculated Spans:", doc['ner'])

            st.markdown("#### Detailed Token Scores")
            tok_idx = st.number_input("Inspect Token Index", min_value=0, max_value=len(doc['tokenized_text'])-1, value=0)
            
            token_word = doc['tokenized_text'][tok_idx]
            scores = doc['scores'][tok_idx]
            
            c1, c2 = st.columns(2)
            c1.metric("Token", token_word)
            c2.write(scores)

if __name__ == "__main__":
    main()