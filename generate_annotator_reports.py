"""
generate_annotator_reports.py

Generates a highly detailed, personalized Markdown report for each annotator.
Features included:
 - MACE Competence Scores
 - Annotation Style Profiling (Radar Charts) with detailed metric interpretations
 - Peer Ranking Boards (Randomized ID peer comparisons)
 - The "AI Blindspot" Hunter (Entities the AI missed but the human caught)
 - Seaborn Disagreement Chart (Where your overruled tags went)
 - Lexical Ambiguities & QC Checks
 - Team Awards / Leaderboards (Gamification)

RUN LOCALLY:
    pip install seaborn matplotlib pandas numpy
    python generate_annotator_reports.py
"""

import os
import json
import math
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from  matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_processing import (
    cwed4eta_process_json_file,
    convert_to_token_spans,
    process_directory_of_json_files,
)

# ============================================================
# 0. CONFIGURATION 
# ============================================================

ANNOTATOR_DIR = "/home/p0l3/RAD/DROP/CLIRENER/ANNOTATORS/"
REFERENT_DIR = "RESULTS/FT_GOLD/"         
OUTPUT_DIR = "REPORTS/ANNOTATOR_REPORTS_DETAILED"

# Precomputed MACE Competence Scores
MACE_SCORES = {
    1: 0.8550, 2: 0.9109, 3: 0.9137, 4: 0.7755, 5: 0.8643, 
    6: 0.8251, 8: 0.8556, 9: 0.8280, 10: 0.8510, 11: 0.8604, 12: 0.8602
}

DATA_CONFIG = {
    1:  {"path": "1/G3_10226.json",    "ids": [4, 1, 5]},
    2:  {"path": "2/G4_5326.json",     "ids": [1, 6]},
    3:  {"path": "3/G4_4326.json",     "ids": [1, 7]},
    4:  {"path": "4/G2_5326_1.json",   "ids": [1, 8]},
    5:  {"path": "5/",                 "ids": [1, 9], "is_dir": True},
    6:  {"path": "6/G1_15126.json",    "ids": [13]},
    8:  {"path": "8/G6_5326.json",     "ids": [12]},
    9:  {"path": "9/G2_5326.json",     "ids": [1, 11]},
    10: {"path": "10/G3_10226_2.json", "ids": [1, 15]},
    11: {"path": "11/G5_21126.json",   "ids": [4, 1, 14]},
    12: {"path": "12/G5_4326.json",    "ids": [4, 1, 2, 3]},
}

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
YOU_COLOR = "#DE4968"
PEER_COLOR = "#CCCCCC"

# ============================================================
# 1. LOADING & SPAN HELPERS
# ============================================================

def safe_load(subpath, annotator_ids, is_directory=False):
    full_path = os.path.join(ANNOTATOR_DIR, subpath)
    if not os.path.exists(full_path): return None
    if is_directory:
        raw = process_directory_of_json_files(full_path, annotator_ids)
    else:
        raw = cwed4eta_process_json_file(full_path, annotator_ids)
    return convert_to_token_spans(raw)

def load_all_annotators():
    data = {}
    for ann_id, cfg in DATA_CONFIG.items():
        if ann_id == 12:
            with open(ANNOTATOR_DIR + "12/" + "G5_4326.json", "r") as f:
                loaded = json.load(f)
        else:
            loaded = safe_load(cfg["path"], cfg["ids"], cfg.get("is_dir", False))
        if loaded: data[ann_id] = loaded
    return data

def load_reference_data():
    if not os.path.exists(REFERENT_DIR): return []
    raw = process_directory_of_json_files(REFERENT_DIR, -1)
    data = convert_to_token_spans(raw)
    return list({d['id']: d for d in data}.values())

def get_expert_mapping():
    """Returns a dictionary: {annotator_id: {set_of_specialty_labels}}"""
    groups = {
        "G1": {"annotators": [5, 6], "labels": {"Asset", "Policy", "Objective", "Method", "Field of Study", "Intellectual Artefact"}},
        "G2": {"annotators": [4, 9], "labels": {"Location", "Geographical Feature", "Body of Water", "Time Period", "Satellite"}},
        "G3": {"annotators": [1, 10], "labels": {"Mathematical Expression", "Measuring Device", "Physical Phenomenon", "Quantity"}},
        "G4": {"annotators": [2, 3], "labels": {"Body Part", "Chemical", "Disease", "Organism", "Ecosystem"}},
        "G5": {"annotators": [11, 12], "labels": {"Energy Source", "Meteorological Phenomenon", "Natural Disaster", "Natural Phenomenon"}},
        "G6": {"annotators": [7, 5, 8], "labels": {"Physical Artefact", "Organization", "Person", "System"}}
    }
    expert_map = defaultdict(set)
    for data in groups.values():
        for ann_id in data["annotators"]:
            expert_map[ann_id].update(data["labels"])
    return expert_map

def _unpack(span):
    if isinstance(span, dict): return span["start"], span["end"], span["label"]
    return span[0], span[1], span[2]

def spans_to_bio_tags(token_list, ner_spans):
    tags = ["O"] * len(token_list)
    for item in ner_spans:
        s, e, l = _unpack(item)
        if s >= len(tags): continue
        tags[s] = f"B-{l}"
        for i in range(s + 1, min(e, len(tags) - 1) + 1):
            tags[i] = f"I-{l}"
    return tags

# ============================================================
# 2. METRICS & ANALYSIS ENGINES
# ============================================================

def calc_style_metrics(docs):
    total_toks = total_ents = adjacency_hits = 0
    all_ents = []
    class_counts = Counter()
    
    for doc in docs:
        toks = doc["tokenized_text"]
        total_toks += len(toks)
        
        spans = sorted(doc["ner"], key=lambda x: _unpack(x)[0])
        for i, span in enumerate(spans):
            s, e, lbl = _unpack(span)
            text = " ".join(toks[s:e+1]).lower()
            all_ents.append({"len": (e - s) + 1, "text": text, "lbl": lbl})
            class_counts[lbl] += 1
            if i < len(spans) - 1:
                next_s = _unpack(spans[i+1])[0]
                if next_s - e <= 2: adjacency_hits += 1

    total_ents = len(all_ents)
    if total_ents == 0:
        return {"saturation": 0, "ttr": 0, "avg_len": 0, "long_pct": 0, "clump_pct": 0, "entropy": 0, "total_ents": 0}

    saturation = (total_ents / total_toks) * 100 if total_toks else 0
    ttr = len(set(e["text"] for e in all_ents)) / total_ents
    avg_len = np.mean([e["len"] for e in all_ents])
    long_pct = sum(1 for e in all_ents if e["len"] >= 5) / total_ents * 100
    clump_pct = (adjacency_hits / total_ents) * 100
    
    probs = [c / total_ents for c in class_counts.values()]
    entropy = (-sum(p * math.log2(p) for p in probs) / math.log2(len(probs))) if len(probs) > 1 else 0

    return {
        "saturation": saturation, "ttr": ttr, "avg_len": avg_len, 
        "long_pct": long_pct, "clump_pct": clump_pct, "entropy": entropy,
        "total_ents": total_ents
    }

def get_ai_stats(docs, ref_docs):
    """Calculates exactly how AI suggestions were handled."""
    dict_ann = {d["id"]: d for d in docs}
    dict_ref = {d["id"]: d for d in ref_docs}
    common = set(dict_ann) & set(dict_ref)
    
    total = exact = deleted = 0
    for doc_id in common:
        ann_spans = dict_ann[doc_id]["ner"]
        ref_spans = dict_ref[doc_id]["ner"]
        total += len(ref_spans)
        
        for r in ref_spans:
            r_s, r_e, r_l = _unpack(r)
            matched_exact = overlapped = False
            for a in ann_spans:
                a_s, a_e, a_l = _unpack(a)
                if max(r_s, a_s) <= min(r_e, a_e):
                    overlapped = True
                    if r_s == a_s and r_e == a_e and r_l == a_l:
                        matched_exact = True
                        break
            if matched_exact: exact += 1
            elif not overlapped: deleted += 1
                
    acc_rate = (exact / total * 100) if total else 0
    del_rate = (deleted / total * 100) if total else 0
    return {"acceptance": acc_rate, "deletion": del_rate}

def get_domain_defender_score(docs, ref_docs, my_expert_labels):
    """Calculates if the annotator scrutinized their own specialty more than other labels."""
    dict_ann = {d["id"]: d for d in docs}
    dict_ref = {d["id"]: d for d in ref_docs}
    common = set(dict_ann) & set(dict_ref)
    
    exp_tot = exp_edits = 0
    non_tot = non_edits = 0
    
    for doc_id in common:
        ann_spans = dict_ann[doc_id]["ner"]
        ref_spans = dict_ref[doc_id]["ner"]
        
        for r in ref_spans:
            r_s, r_e, r_l = _unpack(r)
            accepted_exactly = False
            for a in ann_spans:
                if _unpack(r) == _unpack(a):
                    accepted_exactly = True
                    break
            
            if r_l in my_expert_labels:
                exp_tot += 1
                if not accepted_exactly: exp_edits += 1
            else:
                non_tot += 1
                if not accepted_exactly: non_edits += 1
                
    exp_edit_rate = (exp_edits / exp_tot * 100) if exp_tot else 0
    non_edit_rate = (non_edits / non_tot * 100) if non_tot else 0
    return exp_edit_rate - non_edit_rate  # Positive means they defended their domain harder

def get_consensus_win_rate(ann_id, all_data):
    """How often did the annotator's tag match the majority consensus in overlapping docs?"""
    my_docs = {d["id"]: d for d in all_data[ann_id]}
    total_evals = total_wins = 0
    transitions = Counter()
    
    for doc_id, my_doc in my_docs.items():
        toks = my_doc["tokenized_text"]
        my_tags = spans_to_bio_tags(toks, my_doc["ner"])
        
        peer_tags = []
        for oid, o_docs in all_data.items():
            if oid == ann_id: continue
            o_doc = next((d for d in o_docs if d["id"] == doc_id), None)
            if o_doc: peer_tags.append(spans_to_bio_tags(toks, o_doc["ner"]))
                
        if not peer_tags: continue
        
        for t_idx, my_tag in enumerate(my_tags):
            my_lbl = my_tag[2:] if my_tag != "O" else "O"
            peer_lbls = [pt[t_idx][2:] if pt[t_idx] != "O" else "O" for pt in peer_tags]
            
            # Skip if nobody tagged anything here
            if my_lbl == "O" and all(p == "O" for p in peer_lbls):
                continue
                
            total_evals += 1
            consensus_lbl = Counter(peer_lbls).most_common(1)[0][0] if peer_lbls else "O"
            
            if my_lbl == consensus_lbl:
                total_wins += 1
            elif my_lbl != "O": # Record transition only if they tagged something that was overruled
                transitions[(my_lbl, consensus_lbl)] += 1
                
    win_rate = (total_wins / total_evals * 100) if total_evals else 0
    return win_rate, transitions

def get_ai_blindspots(docs, ref_docs):
    added_counts = Counter()
    dict_ann = {d["id"]: d for d in docs}
    dict_ref = {d["id"]: d for d in ref_docs}
    common = set(dict_ann) & set(dict_ref)

    for doc_id in common:
        ann_spans = dict_ann[doc_id]["ner"]
        ref_spans = dict_ref[doc_id]["ner"]

        for a_span in ann_spans:
            a_s, a_e, a_l = _unpack(a_span)
            overlap = False
            for r_span in ref_spans:
                r_s, r_e, _ = _unpack(r_span)
                if max(a_s, r_s) <= min(a_e, r_e):
                    overlap = True
                    break
            if not overlap:
                added_counts[a_l] += 1
    return added_counts

def get_irbec_ambiguities(docs):
    phrase_to_labels = defaultdict(set)
    for doc in docs:
        toks = doc["tokenized_text"]
        for span in doc["ner"]:
            s, e, lbl = _unpack(span)
            text = " ".join(toks[s:e+1]).lower()
            phrase_to_labels[text].add(lbl)
            
    ambiguities = Counter()
    for text, labels in phrase_to_labels.items():
        if len(labels) > 1:
            pair = " <--> ".join(sorted(list(labels)))
            ambiguities[pair] += 1
    return ambiguities

def check_overlapping_spans(docs):
    issues = []
    for doc in docs:
        spans = doc["ner"]
        for i in range(len(spans)):
            for j in range(i+1, len(spans)):
                s1, e1, l1 = _unpack(spans[i])
                s2, e2, l2 = _unpack(spans[j])
                if max(s1, s2) <= min(e1, e2):
                    toks = doc["tokenized_text"]
                    issues.append({
                        "doc": doc["id"],
                        "t1": f"'{' '.join(toks[s1:e1+1])}' ({l1})",
                        "t2": f"'{' '.join(toks[s2:e2+1])}' ({l2})"
                    })
    return issues

# ============================================================
# 3. PLOTTING ENGINES
# ============================================================

def plot_radar_chart(you_stats, team_stats, out_path):
    metrics = ['saturation', 'ttr', 'avg_len', 'long_pct', 'clump_pct', 'entropy']
    labels = ['Density\n(Ents/100 words)', 'Vocab Diversity\n(TTR)', 'Avg Length\n(Tokens)', 
              'Long Spans\n(>4 words %)', 'Clumping\n(Adjacent %)', 'Class Balance\n(Entropy)']
    
    you_vals, team_vals = [], []
    for m in metrics:
        mx = max(you_stats[m], team_stats[m]) or 1
        you_vals.append(you_stats[m] / mx)
        team_vals.append(team_stats[m] / mx)
        
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    you_vals += you_vals[:1]
    team_vals += team_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, team_vals, color=PEER_COLOR, linewidth=2, linestyle='solid', label='Team Average')
    ax.fill(angles, team_vals, color=PEER_COLOR, alpha=0.25)
    
    ax.plot(angles, you_vals, color=YOU_COLOR, linewidth=2, linestyle='solid', label='You')
    ax.fill(angles, you_vals, color=YOU_COLOR, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_yticklabels([])
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_peer_rankings(you_id, all_stats, ai_stats, peer_map, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    def _draw_bar(ax, title, data_dict, is_percent=False):
        sorted_items = sorted(data_dict.items(), key=lambda x: x[1])
        y_pos = np.arange(len(sorted_items))
        vals = [x[1] for x in sorted_items]
        colors = [YOU_COLOR if x[0] == you_id else PEER_COLOR for x in sorted_items]
        labels = ["You" if x[0] == you_id else peer_map.get(x[0], str(x[0])) for x in sorted_items]
        
        ax.barh(y_pos, vals, color=colors, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_title(title, fontsize=12)
        if is_percent: ax.set_xlabel('%')

    ents = {k: v["total_ents"] for k, v in all_stats.items()}
    _draw_bar(axes[0], "Total Entities Tagged", ents)
    
    active_mace = {k: MACE_SCORES.get(k, 0) for k in all_stats.keys()}
    _draw_bar(axes[1], "MACE Reliability Score (0-1)", active_mace)
    
    ttrs = {k: v["ttr"] for k, v in all_stats.items()}
    _draw_bar(axes[2], "Vocabulary Diversity (Unique/Total)", ttrs)
    
    ai_acc = {k: v["acceptance"] for k, v in ai_stats.items()}
    _draw_bar(axes[3], "Exact AI Suggestion Acceptance Rate", ai_acc, is_percent=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_ai_blindspots(blindspots, out_path):
    if not blindspots: return False
    top = blindspots.most_common(6)
    labels = [x[0] for x in top]
    vals = [x[1] for x in top]
    
    plt.figure(figsize=(8, 4))
    sns.barplot(x=vals, y=labels, hue=labels, palette="magma", legend=False)
    plt.xlabel("Number of Entities Added")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    return True

def plot_sankey_transitions(transitions, out_path):
    """Horizontal bar chart showing where overruled tags went."""
    if not transitions: return False
    top_trans = transitions.most_common(12)
    if not top_trans: return False
    
    top_trans.reverse() # Largest at top
    y_labels, x_vals = [], []
    
    for (src, tgt), v in top_trans:
        t_name = "O (Rejected)" if tgt == "O" else f"{tgt}"
        y_labels.append(f"You: {src} -> Team: {t_name}")
        x_vals.append(v)
        
    plt.figure(figsize=(8, max(4, len(y_labels) * 0.6)))
    sns.barplot(x=x_vals, y=y_labels, hue=y_labels, palette="rocket", legend=False)
    
    plt.xlabel("Number of Times Overruled")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    return True


# ============================================================
# 4. LEADERBOARDS GENERATOR
# ============================================================

def generate_leaderboards(all_stats, ai_stats, win_rates, domain_defense):
    """Calculates top 2 IDs for each Gamification Award."""
    boards = {}
    
    def get_top_2(data_dict, reverse=True):
        return sorted(data_dict.items(), key=lambda x: x[1], reverse=reverse)[:2]

    boards["Skeptic"] = get_top_2({k: v["deletion"] for k, v in ai_stats.items()})
    boards["RubberStamper"] = get_top_2({k: v["acceptance"] for k, v in ai_stats.items()})
    boards["Gatekeeper"] = get_top_2(domain_defense)
    boards["Minimalist"] = get_top_2({k: v["avg_len"] for k, v in all_stats.items()}, reverse=False)
    boards["Maximalist"] = get_top_2({k: v["avg_len"] for k, v in all_stats.items()})
    boards["Oracle"] = get_top_2(win_rates)
    
    return boards


# ============================================================
# 5. MARKDOWN REPORT GENERATOR
# ============================================================

def render_report(ann_id, docs, all_stats, ai_stats, mace_scores, peer_map, leaderboards, irbec, qc, chart_files):
    lines = []
    lines.append(f"# Your Annotation Report: Behind the Scenes\n")
    lines.append(f"_Prepared for Annotator #{ann_id}_\n")
    lines.append("> Thank you for your massive contribution to CliReNER! This report gives you a behind-the-scenes look at your unique annotation style, how you compare to the rest of the anonymized team, and some interesting linguistic patterns you created along the way.\n")

    # --- Section 1: Topline ---
    lines.append("## 🏆 1. Overview & Competence\n")
    lines.append(f"- **Total Sentences Processed:** {len(docs)}")
    lines.append(f"- **Total Entities Tagged:** {all_stats[ann_id]['total_ents']}")
    if ann_id in mace_scores:
        lines.append(f"- **MACE Reliability Score:** **{mace_scores[ann_id]:.4f}**")
        lines.append("  *The MACE score is an unsupervised statistical estimation of reliability (0.0 to 1.0). It measures how often your annotations represent the mathematical 'ground truth' based on team consensus modeling.*\n")
    
    # --- Section 2: Radar Chart ---
    lines.append("## 🕸️ 2. Your Annotation Fingerprint\n")
    lines.append("We tracked 6 advanced style metrics to see how you approach text compared to the team average:\n")
    lines.append(f"![Annotation Fingerprint]({chart_files['radar']})\n")
    
    lines.append("### Understanding the Metrics:")
    lines.append("- **Density (Ents/100 words):** A high score means you are a 'maximalist' who tags many entities per sentence, while a lower score indicates a more conservative approach.")
    lines.append("- **Vocab Diversity (TTR):** A high score means you constantly find unique, novel entity phrases. A lower score means you frequently tag common, recurring keywords.")
    lines.append("- **Avg Length (Tokens):** A high score means you prefer capturing full, descriptive phrases (e.g., 'severe coastal flooding events'), while a lower score means you prefer isolating root concepts (e.g., 'flooding').")
    lines.append("- **Long Spans (>4 words %):** A high score shows you are highly comfortable tagging complex, compound scientific concepts that take many words to describe.")
    lines.append("- **Clumping (Adjacent %):** A high score means you excel at spotting dense enumerations and lists of entities tightly packed together.")
    lines.append("- **Class Balance (Entropy):** A high score means you use a wide, balanced variety of labels. A lower score means your annotations are heavily concentrated in a few specific 'favorite' categories.\n")
    
    # --- Section 3: Peer Comparisons ---
    lines.append("## 📊 3. How You Compare to the Team\n")
    lines.append("This chart shows how your overall volume, reliability (MACE), vocabulary diversity, and AI acceptance rate stack up against the rest of the anonymized team.\n")
    lines.append(f"![Peer Rankings]({chart_files['peers']})\n")

    # --- Section 4: AI Blindspots ---
    lines.append("## 🎯 4. The 'AI Blindspot' Hunter\n")
    if "blindspots" in chart_files:
        lines.append("This chart shows the top entity types you manually added because the AI completely missed them. You are the team's best hunter for these specific concepts!\n")
        lines.append(f"![AI Blindspots]({chart_files['blindspots']})\n")
    else:
        lines.append("*Not enough data to calculate AI blindspots for your assignments.*\n")

    # --- Section 5: Disagreements ---
    lines.append("## ⚔️ 5. Where Your Tags Went When Overruled\n")
    if "sankey" in chart_files:
        lines.append("This chart illustrates what happens when the team consensus overrules your tags. The left side is what you selected, and the right side shows what the team ultimately decided (often 'O (Rejected)', meaning they removed it entirely, or another closely related class).\n")
        lines.append(f"![Disagreements Diagram]({chart_files['sankey']})\n")
    else:
        lines.append("*Not enough peer overlap to calculate a disagreement flow chart.*\n")

    # --- Section 6: Lexical Ambiguities ---
    lines.append("## 🧠 6. Lexical Ambiguities (IRBEC)\n")
    if not irbec:
        lines.append("*You are incredibly consistent! You never used the exact same text string for two different entity types.*\n")
    else:
        lines.append("Sometimes the exact same word acts as different entities depending on the context. Here are the top phrases where you had to make a tough contextual split in your own annotations:\n")
        for pair, cnt in irbec.most_common(4):
            lines.append(f"- **{pair}** ({cnt} occurrences)")
        lines.append("")

    # # --- Section 7: Quality Control ---
    # lines.append("## 🚨 7. Quality Control Flag\n")
    # if not qc:
    #     lines.append("*Clean run! No physically overlapping spans detected in your file.*\n")
    # else:
    #     lines.append(f"**Heads up:** We detected {len(qc)} instance(s) where spans overlap physically (which flat NER doesn't support). Example:\n")
    #     ex = qc[0]
    #     lines.append(f"- {ex['t1']} overlaps with {ex['t2']}\n")

    # --- Section 8: Leaderboards (Gamification) ---
    lines.append("## 🏅 7. The CliReNER Team Awards\n")
    lines.append("To wrap things up, we normalized the statistics across all annotators to find our true specialists. Here is where the team excelled:\n")

    def format_winners(award_key):
        top_2 = leaderboards[award_key]
        winner_texts = []
        for rank, (win_id, val) in enumerate(top_2):
            name = "**You**" if win_id == ann_id else f"*{peer_map.get(win_id, str(win_id))}*"
            medal = "🥇" if rank == 0 else "🥈"
            winner_texts.append(f"{medal} {name} ({val:.1f}%)" if isinstance(val, float) else f"{medal} {name} ({val:.1f})")
        return " | ".join(winner_texts)

    lines.append(f"- **The Ultimate Skeptic** (Highest AI deletion rate): {format_winners('Skeptic')}")
    lines.append(f"- **The Rubber Stamper** (Highest AI exact-match acceptance): {format_winners('RubberStamper')}")
    lines.append(f"- **The Gatekeeper** (Hardest scrutinizer of their own domain expertise): {format_winners('Gatekeeper')}")
    lines.append(f"- **The Minimalist** (Lowest average span length): {format_winners('Minimalist')}")
    lines.append(f"- **The Maximalist** (Highest average span length): {format_winners('Maximalist')}")
    lines.append(f"- **The Oracle** (Highest win-rate when disagreeing with peers): {format_winners('Oracle')}\n")

    return "\n".join(lines)


# ============================================================
# 5. MAIN ORCHESTRATION
# ============================================================

def main():
    print("=== Loading annotator data ===")
    all_data = load_all_annotators()
    ref_data = load_reference_data()
    expert_map = get_expert_mapping()

    print("=== Calculating Universal Metrics & Leaderboards ===")
    all_stats = {}
    ai_stats = {}
    win_rates = {}
    domain_defense = {}
    transitions_map = {}

    for ann_id, docs in all_data.items():
        all_stats[ann_id] = calc_style_metrics(docs)
        ai_stats[ann_id] = get_ai_stats(docs, ref_data)
        
        # Domain defender score
        my_experts = expert_map.get(ann_id, set())
        domain_defense[ann_id] = get_domain_defender_score(docs, ref_data, my_experts)
        
        # Oracle / Consensus
        wr, trans = get_consensus_win_rate(ann_id, all_data)
        win_rates[ann_id] = wr
        transitions_map[ann_id] = trans

    leaderboards = generate_leaderboards(all_stats, ai_stats, win_rates, domain_defense)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ann_id, docs in all_data.items():
        print(f"\n--- Generating Report for Annotator {ann_id} ---")
        ann_dir = os.path.join(OUTPUT_DIR, f"ann_{ann_id}")
        os.makedirs(ann_dir, exist_ok=True)

        # Generate stable, randomized peer mappings specific to this report
        other_ids = [k for k in all_data.keys() if k != ann_id]
        random.seed(ann_id * 42) # Deterministic but unique per annotator
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        random.shuffle(letters) # Shuffle so Annotator 1 isn't always 'A'
        peer_map = {oid: f"Annotator {letters[i]}" for i, oid in enumerate(other_ids)}

        # Complex Metrics
        blindspots = get_ai_blindspots(docs, ref_data)
        irbec = get_irbec_ambiguities(docs)
        qc = check_overlapping_spans(docs)

        # Plotting
        chart_files = {}
        
        radar_path = os.path.join(ann_dir, "radar.png")
        plot_radar_chart(all_stats[ann_id], {k: np.mean([s[k] for s in all_stats.values()]) for k in all_stats[ann_id]}, radar_path)
        chart_files["radar"] = "radar.png"
        
        peer_path = os.path.join(ann_dir, "peers.png")
        plot_peer_rankings(ann_id, all_stats, ai_stats, peer_map, peer_path)
        chart_files["peers"] = "peers.png"

        bs_path = os.path.join(ann_dir, "blindspots.png")
        if plot_ai_blindspots(blindspots, bs_path):
            chart_files["blindspots"] = "blindspots.png"

        sankey_path = os.path.join(ann_dir, "sankey.png")
        if plot_sankey_transitions(transitions_map[ann_id], sankey_path):
            chart_files["sankey"] = "sankey.png"

        # Render Markdown
        report_md = render_report(
            ann_id, docs, all_stats, ai_stats, MACE_SCORES, peer_map, 
            leaderboards, irbec, qc, chart_files
        )

        report_path = os.path.join(ann_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
            
        print(f"  Saved: {report_path}")

    print(f"\n✅ All reports successfully generated in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()