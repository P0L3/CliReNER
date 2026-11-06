"""
generate_annotator_reports.py

Generates a personalised report (Markdown + self-contained HTML, optional PDF)
for every annotator, plus team-level CSVs and plots.

Per-annotator content:
  - "At a glance" takeaways (z-scores against the team) and a values table
  - Annotation fingerprint (radar chart + actual values)
  - Peer comparison (anonymised IDs, incl. MACE and timing)
  - Quality against the final (consensus) dataset: per-label strict / lenient F1,
    specialty-label advantage, error taxonomy (boundary / label / missed / extra),
    label confusions
  - Working with the AI pre-annotations: accept / modify / delete, boundary habits
    (signed edge shifts, most added / trimmed tokens), additions that survived
  - Working pace from Label Studio (lead_time, created_at): speed and fatigue curve
  - Label usage vs team, self-consistency, lexical ambiguities
  - One concrete "contested sentence" with your tags next to the final tags
  - Team awards

Requirements:
    pip install seaborn matplotlib pandas numpy markdown
    pip install weasyprint        # optional, only for --pdf

RUN:
    python generate_annotator_reports.py
    python generate_annotator_reports.py --only 3 5 --pdf
    python generate_annotator_reports.py --package            # USB gift folders in PACKAGES/
"""

import argparse
import base64
import html
import json
import math
import os
import random
import re
import shutil
import tempfile
import warnings
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from dataset_processing import (
    CLIRENER_LABELS_V1,
    convert_to_token_spans,
    cwed4eta_process_json_file,
    process_directory_of_json_files,
)

# ============================================================
# 0. CONFIGURATION
# ============================================================

ANNOTATOR_DIR = "/home/p0l3/RAD/DROP/CLIRENER/ANNOTATORS/"
REFERENT_DIR = "RESULTS/FT_GOLD/"                      # AI pre-annotations shown to experts
GOLD_PATH = ANNOTATOR_DIR + "CONSENSUS/6326.json"      # final consensus dataset
GOLD_IDS = [1]                                         # same priority list as boundary_edit_rate.py
OUTPUT_DIR = "REPORTS/ANNOTATOR_REPORTS_DETAILED"

TIME_FLOOR_S = 300               # minimum threshold (s) above which a sentence may count as interrupted;
                                 # the actual threshold is max(this, Q3 + 3*IQR) of the annotator's own times
MIN_LABEL_SUPPORT = 5            # min gold spans for a label to appear in per-label charts
MIN_ADDITIONS_FOR_SURVIVAL = 10  # min additions before a survival rate is reported / ranked

MACE_SCORES = {
    1: 0.8550, 2: 0.9109, 3: 0.9137, 4: 0.7755, 5: 0.8643,
    6: 0.8251, 8: 0.8556, 9: 0.8280, 10: 0.8510, 11: 0.8604, 12: 0.8602,
}

# "preprocessed": file is already in token-span format (no Label Studio structure, no timing)
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
    12: {"path": "12/G5_4326.json",    "ids": [4, 1, 2, 3], "preprocessed": True},
}

EXPERT_GROUPS = {
    "G1": {"annotators": [5, 6], "labels": {"Asset", "Policy", "Objective", "Method", "Field of Study", "Intellectual Artefact"}},
    "G2": {"annotators": [4, 9], "labels": {"Location", "Geographical Feature", "Body of Water", "Time Period", "Satellite"}},
    "G3": {"annotators": [1, 10], "labels": {"Mathematical Expression", "Measuring Device", "Physical Phenomenon", "Quantity"}},
    "G4": {"annotators": [2, 3], "labels": {"Body Part", "Chemical", "Disease", "Organism", "Ecosystem"}},
    "G5": {"annotators": [11, 12], "labels": {"Energy Source", "Meteorological Phenomenon", "Natural Disaster", "Natural Phenomenon"}},
    "G6": {"annotators": [7, 5, 8], "labels": {"Physical Artefact", "Organization", "Person", "System"}},
}

N_LABELS = len(CLIRENER_LABELS_V1)

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
YOU_COLOR = "#DE4968"
PEER_COLOR = "#CCCCCC"
NEUTRAL_COLOR = "#7A8FA6"

TAX_ORDER = ["correct", "boundary", "type", "both", "missed", "spurious"]
TAX_LABELS = {
    "correct": "Exact match",
    "boundary": "Boundary differs",
    "type": "Label differs",
    "both": "Boundary and label differ",
    "missed": "Missed (in final, not tagged by you)",
    "spurious": "Extra (tagged by you, not in final)",
}
TAX_COLORS = {
    "correct": "#6BAF92", "boundary": "#F2C14E", "type": "#DE4968",
    "both": "#8C2981", "missed": "#5B5F97", "spurious": "#B0B0B0",
}


def _build_label_colors():
    c20, c20b = plt.get_cmap("tab20"), plt.get_cmap("tab20b")
    colors = [c20(i)[:3] for i in range(20)] + [c20b(i)[:3] for i in range(0, 20, 2)]
    return {lab: colors[i % len(colors)] for i, lab in enumerate(sorted(CLIRENER_LABELS_V1))}


LABEL_RGB = _build_label_colors()

NOTES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thank_you_notes.txt")
# Entry N in the file belongs to the N-th annotator in this list.
# Use range(1, 13) instead if you keep a placeholder entry for annotator 7.
NOTE_ORDER = sorted(DATA_CONFIG)


def load_thank_you_notes(path=NOTES_PATH):
    """
    Parses thank_you_notes.txt: entries separated by a blank line, first line = name,
    remaining lines = note. Returns {annotator_id: (name, note)}; {} if the file is missing.
    """
    if not os.path.exists(path):
        print(f"NOTE: {path} not found; using the default intro for every report.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", f.read().strip()) if b.strip()]
    if len(blocks) != len(NOTE_ORDER):
        print(f"WARNING: {len(blocks)} entries in notes file but {len(NOTE_ORDER)} annotators expected.")
    notes = {}
    for ann_id, block in zip(NOTE_ORDER, blocks):
        lines = [l.strip() for l in block.splitlines()]
        notes[ann_id] = (lines[0], " ".join(l for l in lines[1:] if l))
    return notes


# ---------------- Packaging (USB gift) ----------------
PACKAGE_NAME = "AndrijaThanksYou"
PACKAGES_DIR = "PACKAGES"
GAME_HTML = "/home/p0l3/RAD/NER-TOWER/ner-tower/dist/index.html"
PAPER_PDF = "/home/p0l3/RAD/NER-TOWER/ner-tower/dist/s10584-026-04324-z-1.pdf"
EXPORT_DIR = "/home/p0l3/RAD/DROP/CLIRENER/ANNOTATORS/EXPORT/"      # data_<annotator id>.json
LINKS = [
    ("Paper", "https://link.springer.com/article/10.1007/s10584-026-04324-z"),
    ("Other resources related to the paper", "https://langnet.uniri.hr/resources.html"),
    ("Game repository", "https://github.com/P0L3/ner-tower"),
    ("Play the game online", "https://p0l3.github.io/ner-tower/"),
]


def safe_filename(s):
    s = re.sub(r"[^\w\- ]", "", str(s), flags=re.UNICODE).strip().replace(" ", "_")
    return s or "Annotator"


def raw_sources(annotator_dir, ann_id):
    """Source file(s) of the annotator's Label Studio export (a folder for annotators with several files)."""
    cfg = DATA_CONFIG[ann_id]
    src = os.path.join(annotator_dir, cfg["path"])
    if cfg.get("is_dir"):
        if not os.path.isdir(src):
            return []
        return [os.path.join(src, f) for f in sorted(os.listdir(src)) if f.endswith(".json")]
    return [src] if os.path.exists(src) else []


def package_layout(ann_id, name, annotator_dir):
    base = name if name else f"Annotator_{ann_id}"
    return {
        "folder": PACKAGE_NAME,
        "report": f"CliReNER_Report_{safe_filename(base)}.pdf",
        "game": "nertower.html",
        "paper": os.path.basename(PAPER_PDF),
        "raw": [os.path.basename(p) for p in raw_sources(annotator_dir, ann_id)],
        "data": f"data_{ann_id}.json",
    }


def package_tree(info):
    entries = [(1, info["report"], "your personal report (this document)"),
               (1, info["game"], "NER Tower, the game (opens in any web browser)"),
               (1, info["paper"], "the paper based on the data"),
               (1, "README.txt", "short version of these instructions"),
               (1, "raw_data/", "your annotation data (Label Studio export)")]
    entries += [(2, r, "") for r in info["raw"]]
    entries += [(1, "game_data/", "your annotations, ready to load into the game"),
                (2, info["data"], "drag this file into the game")]
    width = max(2 * lvl + len(n) for lvl, n, _ in entries) + 3
    lines = [info["folder"] + "/"]
    for lvl, n, d in entries:
        left = "  " * lvl + n
        lines.append(left.ljust(width) + d if d else left)
    return "\n".join(lines)


def play_steps(info):
    return [
        f"Double-click {info['game']}. It opens in your web browser, needs no installation and works offline.",
        "'Classic Play' uses the CliReNER Gold dataset; 'Custom Seed Play' and 'Weekly challenge' are also available from the same menu.",
        f"To play with your own annotations: click 'Change dataset', then drag and drop game_data/{info['data']} "
        "into the window. The game then uses the sentences and entities you annotated.",
    ]


def package_readme(info, name):
    out = [f"CliReNER thank-you package{' for ' + name if name else ''}", "", "CONTENTS", package_tree(info), "",
           "HOW TO PLAY NER TOWER"]
    out += [f"{i}. {t}" for i, t in enumerate(play_steps(info), 1)]
    out += ["", "OTHER FILES",
            f"- {info['report']}: your personal report with statistics about your annotation work.",
            f"- {info['paper']}: the paper published on the basis of the data.",
            "- raw_data/: the data files behind your annotations; .json files open in any text editor.",
            "", "LINKS"]
    out += [f"- {label}: {url}" for label, url in LINKS]
    return "\n".join(out) + "\n"


def build_package(ann_id, name, pdf_src, info, annotator_dir, packages_dir):
    pkg = os.path.join(packages_dir, f"ann_{ann_id}", info["folder"])
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    os.makedirs(os.path.join(pkg, "raw_data"))
    os.makedirs(os.path.join(pkg, "game_data"))

    def put(src, rel):
        if not os.path.exists(src):
            print(f"  WARNING: missing {src}")
            return
        shutil.copy2(src, os.path.join(pkg, rel))

    put(pdf_src, info["report"])
    put(GAME_HTML, info["game"])
    put(PAPER_PDF, info["paper"])
    for src in raw_sources(annotator_dir, ann_id):
        put(src, os.path.join("raw_data", os.path.basename(src)))
    put(os.path.join(EXPORT_DIR, info["data"]), os.path.join("game_data", info["data"]))
    with open(os.path.join(pkg, "README.txt"), "w", encoding="utf-8") as f:
        f.write(package_readme(info, name))
    return pkg



# ============================================================
# 1. LOADING
# ============================================================

def index_docs(docs):
    return {d["id"]: d for d in docs if "id" in d}


def safe_load(annotator_dir, subpath, annotator_ids, is_directory=False):
    full_path = os.path.join(annotator_dir, subpath)
    if not os.path.exists(full_path):
        return None
    if is_directory:
        raw = process_directory_of_json_files(full_path, annotator_ids)
    else:
        raw = cwed4eta_process_json_file(full_path, annotator_ids)
    return convert_to_token_spans(raw)


def load_timing(annotator_dir, subpath, annotator_ids, is_directory=False):
    """
    Pulls lead_time / created_at from the Label Studio export for the SAME annotation
    that cwed4eta_process_json_file selects (first matching id in the priority list).
    Returns {doc_id: {"lead_time": float|None, "created_at": str|None}}.
    """
    full = os.path.join(annotator_dir, subpath)
    if not os.path.exists(full):
        return {}
    if is_directory:
        files = [os.path.join(full, f) for f in sorted(os.listdir(full)) if f.endswith("json")]
    else:
        files = [full]
    out = {}
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        for task in tasks:
            by_id = {a.get("completed_by"): a for a in (task.get("annotations") or [])}
            chosen = next((by_id[i] for i in annotator_ids if i in by_id), None)
            if chosen is None:
                continue
            data = task.get("data", {})
            doc_id = f"{data.get('paper_id')}-{data.get('sentence_id')}"
            lt = chosen.get("lead_time")
            out[doc_id] = {
                "lead_time": float(lt) if lt is not None else None,
                "created_at": chosen.get("created_at") or chosen.get("updated_at"),
            }
    return out


def load_all_annotators(annotator_dir):
    docs, timing = {}, {}
    for ann_id, cfg in DATA_CONFIG.items():
        if cfg.get("preprocessed"):
            p = os.path.join(annotator_dir, cfg["path"])
            if not os.path.exists(p):
                continue
            with open(p, "r") as f:
                loaded = json.load(f)
            t = {}
        else:
            loaded = safe_load(annotator_dir, cfg["path"], cfg["ids"], cfg.get("is_dir", False))
            t = load_timing(annotator_dir, cfg["path"], cfg["ids"], cfg.get("is_dir", False))
        if loaded:
            docs[ann_id] = index_docs(loaded)
            timing[ann_id] = t
    return docs, timing


def load_reference_data(referent_dir):
    if not os.path.exists(referent_dir):
        return {}
    raw = process_directory_of_json_files(referent_dir, -1)
    return index_docs(convert_to_token_spans(raw))


def load_gold(gold_path):
    if not os.path.exists(gold_path):
        print(f"WARNING: gold file not found at {gold_path}; gold-based sections will be skipped.")
        return {}
    return index_docs(convert_to_token_spans(cwed4eta_process_json_file(gold_path, GOLD_IDS)))


def get_expert_mapping():
    expert_map = defaultdict(set)
    for data in EXPERT_GROUPS.values():
        for ann_id in data["annotators"]:
            expert_map[ann_id].update(data["labels"])
    return expert_map


# ============================================================
# 2. SPAN HELPERS
# ============================================================

def _unpack(span):
    if isinstance(span, dict):
        return span["start"], span["end"], span["label"]
    return span[0], span[1], span[2]


def spans_of(doc):
    return [tuple(_unpack(s)) for s in doc["ner"]]


def _overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def _same_tokens(d1, d2):
    return len(d1["tokenized_text"]) == len(d2["tokenized_text"])


def _ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def mean_of(xs):
    xs = [x for x in xs if _ok(x)]
    return float(np.mean(xs)) if xs else None


def prf(tp_p, n_p, tp_g, n_g):
    if n_p == 0 and n_g == 0:
        return float("nan"), float("nan"), float("nan")
    p = tp_p / n_p if n_p else 0.0
    r = tp_g / n_g if n_g else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def match_to_reference(ann, ref, tokens):
    """
    For each reference span: accepted (exact boundaries), modified (overlap, different
    boundaries) or deleted (no overlap). Same partition logic as boundary_edit_rate.py.
    Returns a list parallel to `ref` of (kind, record|None).
    """
    out = []
    for r in ref:
        cands = [a for a in ann if _overlaps(a, r)]
        if not cands:
            out.append(("deleted", None))
            continue
        exact = [c for c in cands if c[0] == r[0] and c[1] == r[1]]
        a = exact[0] if exact else max(cands, key=lambda c: min(c[1], r[1]) - max(c[0], r[0]))
        added, trimmed = [], []
        if a[0] < r[0]:
            added += tokens[a[0]:r[0]]
        elif a[0] > r[0]:
            trimmed += tokens[r[0]:a[0]]
        if a[1] > r[1]:
            added += tokens[r[1] + 1:a[1] + 1]
        elif a[1] < r[1]:
            trimmed += tokens[a[1] + 1:r[1] + 1]
        out.append(("accepted" if exact else "modified", {
            "left": a[0] - r[0], "right": a[1] - r[1],
            "added": [t.lower() for t in added], "trimmed": [t.lower() for t in trimmed],
        }))
    return out


def align_spans(ann, gold):
    """
    One-to-one alignment of annotator spans to gold spans, returning (kind, ann|None, gold|None).
    Kinds: correct, type (same extent, other label), boundary (same label, other extent),
    both (overlap, other label and extent), missed, spurious. Leftover annotator spans that
    overlap gold (e.g. a split span) are counted as 'boundary'.
    """
    used_a, used_g, out = set(), set(), []
    for gi, g in enumerate(gold):
        for ai, a in enumerate(ann):
            if ai not in used_a and a == g:
                used_a.add(ai); used_g.add(gi); out.append(("correct", a, g)); break
    for gi, g in enumerate(gold):
        if gi in used_g:
            continue
        for ai, a in enumerate(ann):
            if ai not in used_a and a[0] == g[0] and a[1] == g[1]:
                used_a.add(ai); used_g.add(gi); out.append(("type", a, g)); break
    for same_label, kind in ((True, "boundary"), (False, "both")):
        for gi, g in enumerate(gold):
            if gi in used_g:
                continue
            best, best_ov = None, 0
            for ai, a in enumerate(ann):
                if ai in used_a or (same_label and a[2] != g[2]):
                    continue
                ov = min(a[1], g[1]) - max(a[0], g[0]) + 1
                if ov > best_ov:
                    best, best_ov = ai, ov
            if best is not None:
                used_a.add(best); used_g.add(gi); out.append((kind, ann[best], g))
    for gi, g in enumerate(gold):
        if gi not in used_g:
            out.append(("missed", None, g))
    for ai, a in enumerate(ann):
        if ai not in used_a:
            out.append(("boundary" if any(_overlaps(a, g) for g in gold) else "spurious", a, None))
    return out


# ============================================================
# 3. METRIC ENGINES
# ============================================================

def calc_style_metrics(doc_list):
    total_toks = adjacency_hits = 0
    all_ents, class_counts = [], Counter()
    for doc in doc_list:
        toks = doc["tokenized_text"]
        total_toks += len(toks)
        spans = sorted(spans_of(doc), key=lambda x: x[0])
        for i, (s, e, lbl) in enumerate(spans):
            all_ents.append({"len": e - s + 1, "text": " ".join(toks[s:e + 1]).lower()})
            class_counts[lbl] += 1
            if i < len(spans) - 1 and spans[i + 1][0] - e <= 2:
                adjacency_hits += 1
    n = len(all_ents)
    if n == 0:
        return {"saturation": 0, "ttr": 0, "avg_len": 0, "long_pct": 0, "clump_pct": 0,
                "entropy": 0, "total_ents": 0, "labels_used": 0}
    probs = [c / n for c in class_counts.values()]
    # Normalised by log2(all labels), so using many labels evenly scores high (breadth),
    # not just evenness among the few labels that happened to be used.
    entropy = -sum(p * math.log2(p) for p in probs) / math.log2(N_LABELS)
    return {
        "saturation": n / total_toks * 100 if total_toks else 0,
        "ttr": len({e["text"] for e in all_ents}) / n,
        "avg_len": float(np.mean([e["len"] for e in all_ents])),
        "long_pct": sum(1 for e in all_ents if e["len"] >= 5) / n * 100,
        "clump_pct": adjacency_hits / n * 100,
        "entropy": entropy,
        "total_ents": n,
        "labels_used": len(class_counts),
    }


def ai_interaction(ann_docs, ref_docs, expert_labels):
    """
    How the annotator handled AI pre-annotations. The reference was shown as spans only,
    so 'accepted' means exact boundaries (label is not required to match).
    """
    ids = [i for i in ann_docs if i in ref_docs and _same_tokens(ann_docs[i], ref_docs[i])]
    if not ids:
        return None
    res = {"total": 0, "accepted": 0, "modified": 0, "deleted": 0, "added": 0, "shifts": [],
           "additions": [], "per_doc": {}, "exp_total": 0, "exp_edits": 0, "non_total": 0,
           "non_edits": 0, "n_docs": len(ids)}
    for did in ids:
        A, R = spans_of(ann_docs[did]), spans_of(ref_docs[did])
        toks = ann_docs[did]["tokenized_text"]
        d = Counter()
        for (kind, rec), r in zip(match_to_reference(A, R, toks), R):
            d[kind] += 1
            res[kind] += 1
            res["total"] += 1
            if rec:
                res["shifts"].append(rec)
            bucket = "exp" if r[2] in expert_labels else "non"
            res[f"{bucket}_total"] += 1
            if kind != "accepted":
                res[f"{bucket}_edits"] += 1
        for a in A:
            if not any(_overlaps(a, r) for r in R):
                res["added"] += 1
                res["additions"].append((did, a))
        res["per_doc"][did] = (len(R), d["accepted"], d["deleted"])
    return res


def domain_defense(ai):
    """Edit rate on specialty labels minus edit rate on other labels (percentage points)."""
    if not ai or ai["exp_total"] < 5 or ai["non_total"] < 5:
        return None
    return ai["exp_edits"] / ai["exp_total"] * 100 - ai["non_edits"] / ai["non_total"] * 100


def additions_survival(additions, gold_docs):
    """Of the spans the annotator added (no AI counterpart), how many ended up in the final dataset."""
    out = {"n": 0, "survived": 0, "survived_strict": 0, "by_label": Counter(), "surv_by_label": Counter()}
    for did, (s, e, l) in additions:
        g_doc = gold_docs.get(did)
        if not g_doc:
            continue
        G = spans_of(g_doc)
        out["n"] += 1
        out["by_label"][l] += 1
        if (s, e, l) in G:
            out["survived_strict"] += 1
        if any(gl == l and gs <= e and s <= ge for gs, ge, gl in G):
            out["survived"] += 1
            out["surv_by_label"][l] += 1
    return out if out["n"] else None


def gold_scores(counts, labels=None):
    tot = Counter()
    for lab, c in counts.items():
        if labels is None or lab in labels:
            tot.update(c)
    return {
        "strict": prf(tot["strict_tp_pred"], tot["ann"], tot["strict_tp_gold"], tot["gold"]),
        "lenient": prf(tot["len_tp_pred"], tot["ann"], tot["len_tp_gold"], tot["gold"]),
        "n_gold": tot["gold"], "n_ann": tot["ann"],
    }


def evaluate_vs_gold(ann_docs, gold_docs):
    ids = [i for i in ann_docs if i in gold_docs and _same_tokens(ann_docs[i], gold_docs[i])]
    if not ids:
        return None
    counts = defaultdict(Counter)
    tax, conf, shifts, contested = Counter(), Counter(), [], []
    n_gold = n_ann = 0
    for did in ids:
        A, G = spans_of(ann_docs[did]), spans_of(gold_docs[did])
        toks = ann_docs[did]["tokenized_text"]
        n_gold += len(G); n_ann += len(A)
        Aset, Gset = set(A), set(G)
        for a in A:
            c = counts[a[2]]
            c["ann"] += 1
            c["strict_tp_pred"] += a in Gset
            c["len_tp_pred"] += any(g[2] == a[2] and _overlaps(a, g) for g in G)
        for g in G:
            c = counts[g[2]]
            c["gold"] += 1
            c["strict_tp_gold"] += g in Aset
            c["len_tp_gold"] += any(a[2] == g[2] and _overlaps(a, g) for a in A)
        bad = 0
        for kind, a, g in align_spans(A, G):
            tax[kind] += 1
            if kind != "correct":
                bad += 1
            if kind in ("type", "both"):
                conf[(a[2], g[2])] += 1
            elif kind == "spurious":
                conf[(a[2], "(no entity)")] += 1
            elif kind == "missed":
                conf[("(no entity)", g[2])] += 1
        for kind, rec in match_to_reference(A, G, toks):
            if rec:
                shifts.append(rec)
        contested.append((bad, len(toks), did))
    cands = [c for c in contested if c[0] >= 2 and 8 <= c[1] <= 70]
    best = max(cands, key=lambda c: (c[0], -c[1]))[2] if cands else None
    return {"counts": dict(counts), "tax": tax, "conf": conf, "shifts": shifts, "n_docs": len(ids),
            "n_gold": n_gold, "n_ann": n_ann, "contested": best}


def label_table(counts, min_support):
    rows = []
    for lab, c in counts.items():
        if c["gold"] >= min_support:
            sc = gold_scores({lab: c})
            rows.append({"label": lab, "strict": sc["strict"][2], "lenient": sc["lenient"][2], "n_gold": c["gold"]})
    return pd.DataFrame(rows, columns=["label", "strict", "lenient", "n_gold"])


def label_rates(doc_list):
    c = Counter()
    for d in doc_list:
        for s in spans_of(d):
            c[s[2]] += 1
    n = max(len(doc_list), 1)
    return {k: v / n * 100 for k, v in c.items()}


def timing_stats(timing, docs, ai_per_doc):
    if not timing:
        return None
    rows = []
    for did, t in timing.items():
        if did not in docs or t.get("lead_time") is None:
            continue
        tot, acc, dele = ai_per_doc.get(did, (0, 0, 0))
        rows.append({"lt": t["lead_time"], "ts": pd.to_datetime(t.get("created_at"), utc=True, errors="coerce"),
                     "n_ents": len(docs[did]["ner"]), "sug": tot, "acc": acc, "dele": dele})
    if len(rows) < 20:
        return None
    df = pd.DataFrame(rows).sort_values("ts", kind="stable", na_position="last").reset_index(drop=True)
    # Median / quartiles use ALL sentences (robust to outliers, no pace-dependent censoring).
    q1, med, q3 = (float(df["lt"].quantile(q)) for q in (0.25, 0.5, 0.75))
    # Interrupted = far outlier relative to the annotator's OWN pace (Tukey outer fence),
    # never below TIME_FLOOR_S. Used only for sums (active time, entities/min) and the fatigue curve.
    threshold = max(q3 + 3 * (q3 - q1), TIME_FLOOR_S)
    valid = df[df["lt"] <= threshold].reset_index(drop=True)
    if len(valid) < 10:
        return None
    valid["bin"] = np.minimum(np.arange(len(valid)) * 5 // len(valid), 4)
    bins = valid.groupby("bin").apply(lambda g: pd.Series({
        "median_lt": g["lt"].median(),
        "acc": g["acc"].sum() / g["sug"].sum() * 100 if g["sug"].sum() else np.nan,
        "dele": g["dele"].sum() / g["sug"].sum() * 100 if g["sug"].sum() else np.nan,
    }))
    first, last = df["ts"].min(), df["ts"].max()
    return {
        "n": len(df), "n_valid": len(valid), "n_interrupted": len(df) - len(valid),
        "median": med, "q1": q1, "q3": q3, "threshold": float(threshold),
        "hours": float(valid["lt"].sum() / 3600),
        "ents_per_min": float(valid["n_ents"].sum() / (valid["lt"].sum() / 60)) if valid["lt"].sum() else None,
        "first": first.strftime("%Y-%m-%d") if pd.notna(first) else None,
        "last": last.strftime("%Y-%m-%d") if pd.notna(last) else None,
        "active_days": int(valid["ts"].dt.date.nunique()) if valid["ts"].notna().any() else None,
        "bins": bins,
    }


def get_irbec_ambiguities(doc_list):
    phrase_labels = defaultdict(Counter)
    for doc in doc_list:
        toks = doc["tokenized_text"]
        for s, e, lbl in spans_of(doc):
            phrase_labels[" ".join(toks[s:e + 1]).lower()][lbl] += 1
    pairs, examples = Counter(), defaultdict(list)
    for phrase, labs in phrase_labels.items():
        if len(labs) > 1:
            pair = " vs. ".join(sorted(labs))
            pairs[pair] += 1
            examples[pair].append((sum(labs.values()), phrase))
    ex = {p: [ph for _, ph in sorted(v, reverse=True)[:2]] for p, v in examples.items()}
    return pairs, ex


def find_possible_misses(doc_list, min_tagged=2, max_len=8, top=5):
    """Strings you tagged at least twice that also appear, untagged, elsewhere in your sentences."""
    tagged, label_of = Counter(), defaultdict(Counter)
    for doc in doc_list:
        toks = [t.lower() for t in doc["tokenized_text"]]
        for s, e, l in spans_of(doc):
            key = tuple(toks[s:e + 1])
            if 0 < len(key) <= max_len:
                tagged[key] += 1
                label_of[key][l] += 1
    cand = {k for k, c in tagged.items()
            if c >= min_tagged and len(" ".join(k)) >= 3 and not all(re.fullmatch(r"\W+", t) for t in k)}
    if not cand:
        return None
    lengths = sorted({len(k) for k in cand})
    tag_occ, untag_occ, example = Counter(), Counter(), {}
    for doc in doc_list:
        toks = [t.lower() for t in doc["tokenized_text"]]
        spans = [(s, e) for s, e, _ in spans_of(doc)]
        for L in lengths:
            for i in range(len(toks) - L + 1):
                key = tuple(toks[i:i + L])
                if key in cand:
                    j = i + L - 1
                    if any(s <= j and i <= e for s, e in spans):
                        tag_occ[key] += 1
                    else:
                        untag_occ[key] += 1
                        example.setdefault(key, (doc["tokenized_text"], i, j))
    total = sum(tag_occ.values()) + sum(untag_occ.values())
    score = sum(tag_occ.values()) / total * 100 if total else None
    lst = []
    # keep phrases that are genuinely mixed (>=20% tagged), not ubiquitous words tagged by accident
    mixed = [(k, n) for k, n in untag_occ.most_common() if tag_occ[k] / (tag_occ[k] + n) >= 0.2]
    for key, n_un in mixed[:top]:
        lst.append({"text": " ".join(key), "label": label_of[key].most_common(1)[0][0],
                    "tagged": tag_occ[key], "untagged": n_un, "example": example[key]})
    return {"score": score, "n_strings": len(cand), "top": lst}


def count_overlapping_docs(doc_list):
    n = 0
    for doc in doc_list:
        sp = spans_of(doc)
        if any(_overlaps(sp[i], sp[j]) for i in range(len(sp)) for j in range(i + 1, len(sp))):
            n += 1
    return n


def analyze_annotator(ann_id, docs, ref_docs, gold_docs, timing, expert_labels):
    doc_list = list(docs.values())
    A = {"id": ann_id, "n_docs": len(doc_list), "expert_labels": expert_labels}
    A["style"] = calc_style_metrics(doc_list)
    A["ai"] = ai_interaction(docs, ref_docs, expert_labels)
    A["domain_defense"] = domain_defense(A["ai"])
    A["gold"] = evaluate_vs_gold(docs, gold_docs) if gold_docs else None
    A["survival"] = additions_survival(A["ai"]["additions"], gold_docs) if (A["ai"] and gold_docs) else None
    A["label_rates"] = label_rates(doc_list)
    A["timing"] = timing_stats(timing, docs, A["ai"]["per_doc"] if A["ai"] else {})
    A["irbec"] = get_irbec_ambiguities(doc_list)
    A["misses"] = find_possible_misses(doc_list)
    A["overlap_docs"] = count_overlapping_docs(doc_list)
    return A


def flatten_metrics(A, mace):
    m = dict(A["style"])
    m["mace"] = mace.get(A["id"])
    ai = A["ai"]
    if ai and ai["total"]:
        m["acceptance"] = ai["accepted"] / ai["total"] * 100
        m["modification"] = ai["modified"] / ai["total"] * 100
        m["deletion"] = ai["deleted"] / ai["total"] * 100
    m["domain_defense"] = A["domain_defense"]
    g = A["gold"]
    if g:
        sc = gold_scores(g["counts"])
        m["strict_f1"] = sc["strict"][2] * 100
        m["lenient_f1"] = sc["lenient"][2] * 100
        spec = gold_scores(g["counts"], A["expert_labels"])
        other = gold_scores(g["counts"], set(g["counts"]) - set(A["expert_labels"]))
        if spec["n_gold"] >= 10 and other["n_gold"] >= 10:
            m["domain_gap"] = (spec["strict"][2] - other["strict"][2]) * 100
    s = A["survival"]
    if s and s["n"] >= MIN_ADDITIONS_FOR_SURVIVAL:
        m["survival"] = s["survived"] / s["n"] * 100
    t = A["timing"]
    if t:
        m["median_sec"] = t["median"]
        m["ents_per_min"] = t["ents_per_min"]
    return {k: v for k, v in m.items() if _ok(v)}


# ============================================================
# 4. TAKEAWAYS, TABLE SPECS, AWARDS
# ============================================================

TABLE_SPECS = [
    ("saturation", "Density (entities / 100 tokens)", "{:.2f}"),
    ("ttr", "Vocabulary diversity (TTR)", "{:.2f}"),
    ("avg_len", "Average span length (tokens)", "{:.2f}"),
    ("long_pct", "Long spans (5+ tokens, %)", "{:.1f}"),
    ("clump_pct", "Clumping (adjacent entities, %)", "{:.1f}"),
    ("entropy", "Label breadth (entropy / log2 of 28 labels)", "{:.2f}"),
    ("acceptance", "AI suggestions kept as proposed (%)", "{:.1f}"),
    ("deletion", "AI suggestions rejected (%)", "{:.1f}"),
    ("strict_f1", "Strict F1 vs final dataset (%)", "{:.1f}"),
    ("survival", "Added entities retained in final dataset (%)", "{:.1f}"),
    ("median_sec", "Median seconds per sentence", "{:.1f}"),
]

TAKEAWAYS = {
    "saturation": ("{:.2f} entities / 100 tokens",
                   "You tag more entities per sentence than most of the team.",
                   "You tag fewer entities per sentence than most of the team, a more conservative style."),
    "ttr": ("{:.2f}", "You tag a more varied set of phrases than most of the team.",
            "You tag more recurring phrases than most of the team."),
    "avg_len": ("{:.2f} tokens", "You tend to capture longer, more descriptive spans.",
                "You tend to isolate shorter core concepts."),
    "clump_pct": ("{:.1f}%", "You pick up dense lists of adjacent entities more than most.",
                  "Adjacent entity clusters are less common in your annotations than in the team's."),
    "entropy": ("{:.2f}", "You use a broader range of labels than most of the team.",
                "Your annotations concentrate on a narrower set of labels than the team's."),
    "acceptance": ("{:.1f}%", "You kept a larger share of AI suggestions exactly as proposed than most.",
                   "You edited or removed AI suggestions more often than most of the team."),
    "deletion": ("{:.1f}%", "You rejected AI suggestions outright more often than most.",
                 "You rarely rejected AI suggestions outright compared with the team."),
    "strict_f1": ("{:.1f}%", "Your spans match the final dataset exactly more often than the team's typical.",
                  "Your exact-match agreement with the final dataset is below the team's typical, which often reflects "
                  "harder label sets as much as quality; see the per-label breakdown."),
    "survival": ("{:.1f}%", "Entities you added beyond the AI suggestions were retained in the final dataset more often than most.",
                 "A smaller share of the entities you added beyond the AI suggestions made it into the final dataset."),
    "median_sec": ("{:.1f} s", "You spend more time per sentence than most of the team.",
                   "You work faster per sentence than most of the team."),
    "domain_gap": ("{:+.1f} pp", "You agree with the final dataset noticeably better on your specialty labels than on the others.",
                   "Your agreement on specialty labels is no better than on the other labels."),
}

AWARDS = [
    ("Skeptic", "The Ultimate Skeptic", "highest rate of rejecting AI suggestions outright", "deletion", True, "%"),
    ("RubberStamper", "The Rubber Stamper", "highest share of AI suggestions kept with exact boundaries", "acceptance", True, "%"),
    ("Gatekeeper", "The Gatekeeper", "edits AI suggestions on own specialty labels more than on other labels", "domain_defense", True, " pp"),
    ("Minimalist", "The Minimalist", "lowest average span length", "avg_len", False, " tokens"),
    ("Maximalist", "The Maximalist", "highest average span length", "avg_len", True, " tokens"),
    ("Oracle", "The Oracle", "closest to the final dataset (strict F1)", "strict_f1", True, "% F1"),
    ("Scout", "The Scout", "added entities that most often survived into the final dataset", "survival", True, "%"),
]


def rank_of(key, ann_id, metrics):
    vals = {k: m[key] for k, m in metrics.items() if key in m}
    if ann_id not in vals or len(vals) < 2:
        return None
    r = 1 + sum(1 for v in vals.values() if v > vals[ann_id])
    return r, len(vals)


def build_takeaways(ann_id, metrics):
    out = []
    scored = []
    for key, (fmt, hi, lo) in TAKEAWAYS.items():
        vals = [m[key] for m in metrics.values() if key in m]
        if ann_id not in metrics or key not in metrics[ann_id] or len(vals) < 4:
            continue
        sd = float(np.std(vals, ddof=1))
        if sd == 0:
            continue
        mu = float(np.mean(vals))
        v = metrics[ann_id][key]
        scored.append(((v - mu) / sd, key, v, mu))
    scored.sort(key=lambda x: -abs(x[0]))
    for z, key, v, mu in scored[:4]:
        if abs(z) < 0.8 or abs(v - mu) < 0.05 * abs(mu):
            continue
        fmt, hi, lo = TAKEAWAYS[key]
        rk = rank_of(key, ann_id, metrics)
        out.append(f"{hi if z > 0 else lo} (you: {fmt.format(v)}, team mean: {fmt.format(mu)}; "
                   f"rank {rk[0]} of {rk[1]}, 1 = highest)")
    if not out:
        out.append("Your profile sits close to the team average on all tracked metrics.")
    return out


def ordinal(n):
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def generate_leaderboards(metrics):
    boards = {}
    for key, title, desc, mkey, reverse, unit in AWARDS:
        items = [(k, m[mkey]) for k, m in metrics.items() if mkey in m]
        items.sort(key=lambda x: x[1], reverse=reverse)
        boards[key] = {"title": title, "desc": desc, "items": items[:2], "unit": unit}
    return boards


# ============================================================
# 5. PLOTS
# ============================================================

def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_radar_chart(you, team, out_path):
    metrics = ["saturation", "ttr", "avg_len", "long_pct", "clump_pct", "entropy"]
    labels = ["Density\n(Ents/100 tokens)", "Vocab Diversity\n(TTR)", "Avg Length\n(Tokens)",
              "Long Spans\n(5+ tokens %)", "Clumping\n(Adjacent %)", "Label Breadth\n(Entropy)"]
    yv, tv = [], []
    for m in metrics:
        mx = max(you[m], team[m]) or 1
        yv.append(you[m] / mx)
        tv.append(team[m] / mx)
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    yv += yv[:1]; tv += tv[:1]; ang += ang[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(ang, tv, color=PEER_COLOR, linewidth=2, label="Team average")
    ax.fill(ang, tv, color=PEER_COLOR, alpha=0.25)
    ax.plot(ang, yv, color=YOU_COLOR, linewidth=2, label="You")
    ax.fill(ang, yv, color=YOU_COLOR, alpha=0.25)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_yticklabels([])
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save(out_path)


PEER_PANELS = [
    ("total_ents", "Total entities tagged", 1),
    ("mace", "MACE reliability score (0-1)", 1),
    ("ttr", "Vocabulary diversity (unique / total)", 1),
    ("acceptance", "AI suggestions kept as proposed (%)", 1),
    ("strict_f1", "Strict F1 vs final dataset (%)", 1),
    ("median_sec", "Median seconds per sentence", 1),
]


def plot_peer_rankings(you_id, metrics, peer_map, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (key, title, _) in zip(axes.flatten(), PEER_PANELS):
        items = sorted([(k, m[key]) for k, m in metrics.items() if key in m], key=lambda x: x[1])
        if not items:
            ax.axis("off")
            continue
        y = np.arange(len(items))
        ax.barh(y, [v for _, v in items], edgecolor="white",
                color=[YOU_COLOR if k == you_id else PEER_COLOR for k, _ in items])
        ax.set_yticks(y)
        ax.set_yticklabels(["You" if k == you_id else peer_map.get(k, str(k)) for k, _ in items])
        ax.set_title(title, fontsize=11)
    _save(out_path)


def plot_gold_by_label(df, expert_labels, out_path):
    if df.empty:
        return False
    df = df.sort_values("strict", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(df) + 1.5)))
    y = np.arange(len(df))
    h = 0.38
    cols = [YOU_COLOR if l in expert_labels else "#7A7A7A" for l in df["label"]]
    ax.barh(y - h / 2, df["strict"] * 100, h, color=cols, edgecolor="white")
    ax.barh(y + h / 2, df["lenient"] * 100, h, color=cols, alpha=0.4, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l} (n={n})" for l, n in zip(df["label"], df["n_gold"])])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("F1 (%)")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.legend(handles=[Patch(color="#555555", label="Strict (exact span + label)"),
                       Patch(color="#555555", alpha=0.4, label="Lenient (same label, overlapping)"),
                       Patch(color=YOU_COLOR, label="Your specialty labels")],
              loc="lower right", fontsize=8)
    _save(out_path)
    return True


def plot_error_taxonomy(rows, out_path):
    """rows: list of (name, Counter)."""
    rows = [(n, c) for n, c in rows if sum(c.values())]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.9 * len(rows)))
    for i, (name, c) in enumerate(rows):
        total = sum(c.values())
        left = 0
        for k in TAX_ORDER:
            w = c.get(k, 0) / total * 100
            ax.barh(i, w, left=left, color=TAX_COLORS[k], edgecolor="white")
            if w >= 4:
                ax.text(left + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if k in ("type", "both", "missed") else "black")
            left += w
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of aligned items (final-dataset spans + extra spans)")
    ax.legend(handles=[Patch(color=TAX_COLORS[k], label=TAX_LABELS[k]) for k in TAX_ORDER],
              loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8, frameon=False)
    _save(out_path)
    return True


def plot_confusions(conf, out_path, top=10):
    if not conf:
        return False
    items = conf.most_common(top)[::-1]
    cols = sns.color_palette("rocket", len(items))
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.55 * len(items))))
    ax.barh(range(len(items)), [v for _, v in items], color=cols)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f"You: {a}  ->  Final: {b}" for (a, b), _ in items])
    ax.set_xlabel("Number of spans")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    _save(out_path)
    return True


def _shift_dist(recs, key):
    vals = np.clip([r[key] for r in recs], -3, 3)
    return np.array([(vals == v).mean() * 100 for v in range(-3, 4)])


def plot_boundary_shifts(panels, out_path):
    panels = [p for p in panels if p[1]]
    if not panels:
        return False
    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 3.6 * len(panels)), squeeze=False)
    xs = np.arange(-3, 4)
    xt = ["<=-3", "-2", "-1", "0", "+1", "+2", ">=+3"]
    w = 0.38
    for r, (name, you, team) in enumerate(panels):
        for c, (key, title) in enumerate((("left", "Left edge (negative: you started earlier)"),
                                          ("right", "Right edge (positive: you ended later)"))):
            ax = axes[r][c]
            ax.bar(xs - w / 2, _shift_dist(team, key) if team else np.zeros(7), w, color=PEER_COLOR, label="Team")
            ax.bar(xs + w / 2, _shift_dist(you, key), w, color=YOU_COLOR, label="You")
            ax.set_xticks(xs)
            ax.set_xticklabels(xt)
            ax.set_title(f"{name}: {title}", fontsize=9)
            ax.set_xlabel("Shift in tokens")
            ax.set_ylabel("% of matched spans")
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            if r == 0 and c == 1:
                ax.legend()
    _save(out_path)
    return True


def plot_survivors(counter, out_path):
    if not counter:
        return False
    top = counter.most_common(6)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(top)), [v for _, v in top], color=sns.color_palette("magma", len(top)))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([k for k, _ in top])
    ax.set_xlabel("Added entities that made it into the final dataset")
    _save(out_path)
    return True


def plot_label_usage(diff, specialty, out_path, top=12):
    if not diff:
        return False
    items = sorted(diff.items(), key=lambda x: -abs(x[1]))[:top]
    items = sorted(items, key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(items) + 1)))
    ax.barh(range(len(items)), [v for _, v in items],
            color=[YOU_COLOR if v > 0 else NEUTRAL_COLOR for _, v in items])
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([k + (" *" if k in specialty else "") for k, _ in items])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Entities per 100 sentences: you minus team average")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    _save(out_path)
    return True


def _nanmean_stack(arrs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.vstack(arrs), axis=0)


def plot_fatigue(you_bins, team_bins_list, out_path):
    x = np.arange(1, 6)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    if team_bins_list:
        ax.plot(x, _nanmean_stack([b["median_lt"].reindex(range(5)).values for b in team_bins_list]),
                color="#888888", linestyle="--", marker="o", label="Team")
    ax.plot(x, you_bins["median_lt"].reindex(range(5)).values, color=YOU_COLOR, marker="o", label="You")
    ax.set_title("Median seconds per sentence")
    ax = axes[1]
    for col, c, name in (("acc", YOU_COLOR, "kept as proposed"), ("dele", "#4C72B0", "rejected")):
        if team_bins_list:
            ax.plot(x, _nanmean_stack([b[col].reindex(range(5)).values for b in team_bins_list]),
                    color="#888888", linestyle="--", marker="o", alpha=0.7 if col == "acc" else 0.4)
        ax.plot(x, you_bins[col].reindex(range(5)).values, color=c, marker="o", label=f"You: {name}")
    ax.set_title("AI suggestions handled (%), team dashed")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(["1st", "2nd", "3rd", "4th", "5th"])
        ax.set_xlabel("Chronological fifth of your work")
        ax.grid(linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
    _save(out_path)


# ============================================================
# 6. MARKDOWN / HTML RENDERING
# ============================================================

def md_escape(s):
    s = html.escape(str(s), quote=False).replace("|", "/")
    return re.sub(r"([\\`*_\[\]])", r"\\\1", s)


def render_tagged_html(tokens, spans):
    starts = {}
    for s, e, l in spans:
        starts[s] = (max(e, s), l)
    parts, i = [], 0
    while i < len(tokens):
        if i in starts:
            e, l = starts[i]
            r, g, b = [int(255 * c) for c in LABEL_RGB.get(l, (0.7, 0.7, 0.7))]
            txt = " ".join(html.escape(t) for t in tokens[i:e + 1])
            parts.append(f'<span class="ent" style="background:rgba({r},{g},{b},0.35);'
                         f'border-bottom:2px solid rgb({r},{g},{b})">{txt}<sub>{html.escape(l)}</sub></span>')
            i = e + 1
        else:
            parts.append(html.escape(tokens[i]))
            i += 1
    return " ".join(parts)


def example_html(doc_id, ann_docs, gold_docs):
    a, g = ann_docs[doc_id], gold_docs[doc_id]
    toks = a["tokenized_text"]
    return ('<div class="example">'
            f'<p class="ex-title">Your annotation</p><p>{render_tagged_html(toks, spans_of(a))}</p>'
            f'<p class="ex-title">Final dataset</p><p>{render_tagged_html(toks, spans_of(g))}</p></div>')


class MD:
    def __init__(self):
        self.L = []

    def h(self, level, text):
        self.L += [f"{'#' * level} {text}", ""]

    def p(self, text):
        self.L += [text, ""]

    def bullets(self, items):
        self.L += [f"- {i}" for i in items] + [""]

    def table(self, headers, rows):
        self.L.append("| " + " | ".join(headers) + " |")
        self.L.append("|" + "|".join("---" for _ in headers) + "|")
        for r in rows:
            self.L.append("| " + " | ".join(str(c) for c in r) + " |")
        self.L.append("")

    def img(self, alt, src):
        self.L += [f"![{alt}]({src})", ""]

    def raw(self, text):
        self.L += [text, ""]

    def text(self):
        return "\n".join(self.L)


def pct(x, d=1):
    return f"{x:.{d}f}%" if _ok(x) else "n/a"


def _tok_list(counter, n=6):
    items = counter.most_common(n)
    return ", ".join(f"{md_escape(t)} ({c})" for t, c in items) if items else "none"


def _shift_summary(recs, team_recs):
    if not recs:
        return None
    def stats(rs):
        n = len(rs)
        return {"n": n,
                "exact": sum(1 for r in rs if r["left"] == 0 and r["right"] == 0) / n * 100,
                "ext": sum(1 for r in rs if r["left"] < 0 or r["right"] > 0) / n * 100,
                "trim": sum(1 for r in rs if r["left"] > 0 or r["right"] < 0) / n * 100,
                "ml": float(np.mean([r["left"] for r in rs])),
                "mr": float(np.mean([r["right"] for r in rs]))}
    return stats(recs), (stats(team_recs) if team_recs else None)


def render_report(A, ctx):
    ann_id = A["id"]
    metrics, m = ctx["metrics"], ctx["metrics"][ann_id]
    ch, peer_map = ctx["charts"], ctx["peer_map"]
    md = MD()
    name, note = ctx.get("thanks") or (None, None)
    md.h(1, "Your Annotation Report: Behind the Scenes")
    md.p(f"_Prepared for {md_escape(name)}_" if name else f"_Prepared for Annotator #{ann_id}_")
    if note:
        md.p("> " + note)
    else:
        md.p("> Thank you for your substantial contribution to CliReNER.")
    md.p("_This report describes style and contribution. Differences between annotators are expected, "
         "since groups worked on different labels and sentences._")

    md.h(2, "At a glance")
    md.bullets(ctx["takeaways"])

    pk = ctx.get("package")
    if pk:
        md.h(2, "How to use this USB stick")
        md.p("Everything here is yours to keep. Nothing needs to be installed, and no internet connection is needed "
             "(except for the links at the end of this report). The folder contains:")
        md.raw('<pre class="tree">' + html.escape(package_tree(pk)) + "</pre>")
        md.h(3, "Playing NER Tower")
        md.bullets([html.escape(t, quote=False) for t in play_steps(pk)])
        md.h(3, "The paper and the raw data")
        md.bullets([f"**{html.escape(pk['paper'])}** is the paper published on the basis of the annotated data.",
                    "**raw_data/** holds the data files behind your annotations. JSON files open in any text editor; "
                    "most readers can simply ignore this folder.",
                    "**README.txt** repeats these instructions in plain text."])

    # 1 Overview
    md.h(2, "1. Overview and competence")
    items = [f"**Total sentences processed:** {A['n_docs']}",
             f"**Total entities tagged:** {A['style']['total_ents']}",
             f"**Labels used:** {A['style']['labels_used']} of {N_LABELS}"]
    if ann_id in MACE_SCORES:
        items.append(f"**MACE reliability score:** {MACE_SCORES[ann_id]:.4f}")
    md.bullets(items)
    if ann_id in MACE_SCORES:
        md.p("_MACE is an unsupervised estimate of reliability (0 to 1) based on agreement patterns across the team._")

    ai, g, tm = A["ai"], A["gold"], A["timing"]
    n_text = {
        "acceptance": f"{ai['total']} suggestions" if ai else "",
        "deletion": f"{ai['total']} suggestions" if ai else "",
        "strict_f1": f"{g['n_gold']} final spans" if g else "",
        "survival": f"{A['survival']['n']} additions" if A["survival"] else "",
        "median_sec": f"{tm['n_valid']} sentences" if tm else "",
    }
    rows = []
    for key, label, fmt in TABLE_SPECS:
        if key not in m:
            continue
        vals = [x[key] for x in metrics.values() if key in x]
        rk = rank_of(key, ann_id, metrics)
        rows.append([label, fmt.format(m[key]), fmt.format(float(np.mean(vals))),
                     f"{rk[0]} of {rk[1]}" if rk else "n/a",
                     n_text.get(key, f"{A['style']['total_ents']} entities")])
    md.table(["Metric", "You", "Team mean", "Your rank (1 = highest)", "Based on"], rows)

    # 2 Fingerprint
    md.h(2, "2. Your annotation fingerprint")
    md.p("Six style metrics compared with the team average. Each axis is scaled to the larger of the two values, "
         "so check the table above for the actual magnitudes.")
    md.img("Annotation fingerprint", ch["radar"])
    md.h(3, "Understanding the metrics")
    md.bullets([
        "**Density (entities per 100 tokens):** high means many entities per sentence; low means a more conservative approach.",
        "**Vocabulary diversity (TTR):** high means you tag many distinct phrases; low means frequent re-tagging of recurring keywords.",
        "**Average length (tokens):** high means full descriptive phrases (e.g. 'severe coastal flooding events'); low means root concepts (e.g. 'flooding').",
        "**Long spans (5+ tokens):** share of complex, multi-word concepts.",
        "**Clumping:** share of entities directly adjacent to another, typical for enumerations.",
        f"**Label breadth:** label entropy divided by log2({N_LABELS}). High means many labels used fairly evenly; low means a few favourite labels.",
    ])

    # 3 Peers
    md.h(2, "3. How you compare to the team")
    md.p("Anonymised peers. Treat these as context rather than a ranking: groups worked on different labels and "
         "sentence sets, which influences acceptance rates, speed and F1.")
    md.img("Peer comparison", ch["peers"])

    # 4 Gold
    md.h(2, "4. Agreement with the final dataset")
    if not g:
        md.p("_The final dataset was not available for your sentences._")
    else:
        sc = gold_scores(g["counts"])
        md.p(f"Compared on the {g['n_docs']} sentences you annotated ({g['n_gold']} final spans, {g['n_ann']} of yours). "
             "Strict requires exact boundaries and label; lenient requires the same label and any overlap.")
        md.table(["Metric", "Precision", "Recall", "F1"],
                 [["Strict", pct(sc["strict"][0] * 100), pct(sc["strict"][1] * 100), pct(sc["strict"][2] * 100)],
                  ["Lenient", pct(sc["lenient"][0] * 100), pct(sc["lenient"][1] * 100), pct(sc["lenient"][2] * 100)]])
        spec_labels = A["expert_labels"]
        spec = gold_scores(g["counts"], spec_labels)
        other = gold_scores(g["counts"], set(g["counts"]) - set(spec_labels))
        if spec["n_gold"] >= 10 and other["n_gold"] >= 10:
            md.p(f"**Specialty advantage.** On your specialty labels ({', '.join(sorted(l for l in spec_labels if l in g['counts']))}) "
                 f"strict F1 is {spec['strict'][2] * 100:.1f}% (n = {spec['n_gold']} final spans) versus "
                 f"{other['strict'][2] * 100:.1f}% on the other labels (n = {other['n_gold']}).")
        df = label_table(g["counts"], ctx["min_support"])
        if len(df) >= 4:
            top = df.sort_values("strict", ascending=False)
            best, worst = top.head(3), top.tail(3).iloc[::-1]
            md.p("**Strongest labels:** " + "; ".join(f"{md_escape(r.label)} ({r.strict * 100:.0f}%, n={r.n_gold})" for r in best.itertuples())
                 + ".  \n**Most challenging labels:** " + "; ".join(f"{md_escape(r.label)} ({r.strict * 100:.0f}%, n={r.n_gold})" for r in worst.itertuples()) + ".")
        if "gold_label" in ch:
            md.img("Per-label F1", ch["gold_label"])
            md.p(f"_Labels with at least {ctx['min_support']} final spans in your sentences. Your specialty labels are highlighted._")
        md.h(3, "What kind of differences were they?")
        md.p("Every difference from the final dataset falls into one of three kinds: **existence** (a span you missed or added), "
             "**boundary** (right label, different extent) or **label** (right extent, different label); some spans differ in both.")
        if "taxonomy" in ch:
            md.img("Error taxonomy", ch["taxonomy"])
        if "confusions" in ch:
            md.p("Most frequent label-level differences:")
            md.img("Label confusions", ch["confusions"])
        md.p("_The final dataset is built partly from your own annotations, so agreement is somewhat optimistic._")

    # 5 AI
    md.h(2, "5. Working with the AI suggestions")
    if not ai:
        md.p("_No AI pre-annotation overlap was found for your sentences._")
    else:
        t = ctx["team_ai"]
        md.p(f"You saw {ai['total']} AI-suggested spans (team mean in brackets). The suggestions were shown as spans only, "
             "so 'kept' refers to exact boundaries.")
        md.bullets([
            f"**Kept as proposed:** {pct(m.get('acceptance'))} ({pct(t['acceptance'])})",
            f"**Kept but boundaries changed:** {pct(m.get('modification'))} ({pct(t['modification'])})",
            f"**Rejected outright:** {pct(m.get('deletion'))} ({pct(t['deletion'])})",
            f"**Entities you added that the AI had missed:** {ai['added']} ({ai['added'] / max(A['n_docs'], 1) * 100:.1f} per 100 sentences)",
        ])
        md.h(3, "Boundary habits")
        gs = _shift_summary(g["shifts"], ctx["team_shifts_gold"]) if g else None
        ais = _shift_summary(ai["shifts"], ctx["team_shifts_ai"])
        brow = []
        for name, s in (("vs AI suggestions", ais), ("vs final dataset", gs)):
            if s:
                you_s, team_s = s
                tc = lambda k, f: f" (team {f.format(team_s[k])})" if team_s else ""
                brow.append([name, you_s["n"], f"{you_s['ext']:.0f}%{tc('ext', '{:.0f}%')}", f"{you_s['trim']:.0f}%{tc('trim', '{:.0f}%')}",
                             f"{you_s['ml']:+.2f}{tc('ml', '{:+.2f}')}", f"{you_s['mr']:+.2f}{tc('mr', '{:+.2f}')}"])
        if brow:
            md.table(["Compared with", "Matched spans", "Extended an edge", "Trimmed an edge", "Mean left shift", "Mean right shift"], brow)
            md.p("_Negative left shift = you start earlier than the reference; positive right shift = you end later._")
            if ais and ais[1]:
                dl, dr = ais[0]["ml"] - ais[1]["ml"], ais[0]["mr"] - ais[1]["mr"]
                notes = []
                if dr > 0.15: notes.append("you extend right edges more than the team does")
                if dr < -0.15: notes.append("you trim right edges more than the team does")
                if dl < -0.15: notes.append("you extend left edges more than the team does")
                if dl > 0.15: notes.append("you trim left edges more than the team does")
                if notes:
                    md.p("Compared with the team, " + " and ".join(notes) + " when editing AI suggestions.")
            ac, tr = Counter(), Counter()
            for r in ai["shifts"]:
                ac.update(r["added"]); tr.update(r["trimmed"])
            md.bullets([f"**Tokens you most often added to AI spans:** {_tok_list(ac)}",
                        f"**Tokens you most often trimmed from AI spans:** {_tok_list(tr)}"])
            if g and g["shifts"]:
                ac2, tr2 = Counter(), Counter()
                for r in g["shifts"]:
                    ac2.update(r["added"]); tr2.update(r["trimmed"])
                md.bullets([f"**Tokens you added relative to the final dataset:** {_tok_list(ac2)}",
                            f"**Tokens you trimmed relative to the final dataset:** {_tok_list(tr2)}"])
        if "shifts" in ch:
            md.img("Boundary shifts", ch["shifts"])
        md.h(3, "Additions that survived")
        sv = A["survival"]
        if not sv:
            md.p("_No additions could be compared with the final dataset._")
        else:
            md.p(f"Of {sv['n']} entities you added beyond the AI suggestions, **{sv['survived']} ({sv['survived'] / sv['n'] * 100:.0f}%)** "
                 f"appear in the final dataset with the same label ({sv['survived_strict']} with exactly the same boundaries). "
                 "The chart shows the labels where your additions were retained.")
            if "survivors" in ch:
                md.img("Surviving additions", ch["survivors"])

    # 6 Pace
    md.h(2, "6. Working pace")
    if not tm:
        md.p("_Timing data (Label Studio lead_time) was not available for your annotation file._")
    else:
        period = f"{tm['first']} to {tm['last']}" if tm["first"] else "n/a"
        md.bullets([
            f"**Median time per sentence:** {tm['median']:.0f} s (middle half: {tm['q1']:.0f} to {tm['q3']:.0f} s)",
            f"**Active annotation time:** {tm['hours']:.1f} h" + (f" over {tm['active_days']} days" if tm["active_days"] else ""),
            f"**Entities tagged per active minute:** {tm['ents_per_min']:.1f}" if tm["ents_per_min"] else "**Entities per minute:** n/a",
            f"**Period:** {period}",
            f"**Excluded as interrupted (over {tm['threshold']:.0f} s):** {tm['n_interrupted']} of {tm['n']} sentences",
        ])
        md.p("_Active time and entities per minute use only the non-interrupted sentences. Time is Label Studio's per-sentence "
             "timer, summed; it is not wall-clock time. Entities per minute includes entities kept from the AI suggestions, "
             "so it reflects review throughput rather than typing speed._")
        if "fatigue" in ch:
            md.p("Your work split into five chronological parts: time per sentence and how you handled AI suggestions as the work progressed.")
            md.img("Pace over time", ch["fatigue"])

    # 7 Label usage
    md.h(2, "7. Your label usage vs the team")
    if "usage" in ch:
        md.p("Labels where your usage per 100 sentences deviates most from the other annotators (* = your specialty). "
             "Sentence sets differ between annotators, so read this as a tendency.")
        md.img("Label usage", ch["usage"])
    else:
        md.p("_Not enough data._")

    # 8 Consistency
    md.h(2, "8. Consistency and lexical ambiguity")
    ms = A["misses"]
    if ms and ms["score"] is not None:
        md.p(f"**Self-consistency:** across {ms['n_strings']} phrases you tagged at least twice, {ms['score']:.0f}% of their occurrences "
             "in your sentences were tagged. Phrases below appear untagged elsewhere; context may justify this, so treat them as a prompt to look again.")
        rows = []
        for it in ms["top"]:
            toks, i, j = it["example"]
            lo, hi = max(0, i - 8), min(len(toks), j + 9)
            ctxt = " ".join(md_escape(t) for t in toks[lo:i]) + f" **{' '.join(md_escape(t) for t in toks[i:j + 1])}** " + " ".join(md_escape(t) for t in toks[j + 1:hi])
            rows.append([md_escape(it["text"]), md_escape(it["label"]), it["tagged"], it["untagged"], ctxt.strip()])
        if rows:
            md.table(["Phrase", "Usual label", "Tagged", "Untagged", "Example of an untagged occurrence"], rows)
    pairs, ex = A["irbec"]
    if not pairs:
        md.p("_The same text string never received two different labels in your annotations._")
    else:
        md.p("The same text acted as different entity types depending on context. Most frequent label pairs for such strings:")
        md.bullets([f"**{md_escape(pair)}** ({cnt} phrase(s), e.g. {', '.join(md_escape(x) for x in ex[pair])})" for pair, cnt in pairs.most_common(4)])
    if A["overlap_docs"]:
        md.p(f"_Data hygiene: {A['overlap_docs']} of your sentences contain physically overlapping spans, which flat NER cannot represent._")

    # 9 Example
    md.h(2, "9. A sentence worth a second look")
    if ctx.get("example"):
        md.p("The sentence where your tags differed most from the final dataset (among sentences of moderate length):")
        md.raw(ctx["example"])
    else:
        md.p("_No suitable contested sentence found._")

    # 10 Awards
    md.h(2, "10. The CliReNER team awards")
    md.p("Fun facts rather than rankings of quality. A different style is not a better or worse one.")

    def winners(key):
        b = ctx["boards"][key]
        parts = []
        for rank, (wid, val) in enumerate(b["items"], 1):
            name = "**You**" if wid == ann_id else f"*{peer_map.get(wid, str(wid))}*"
            parts.append(f"{ordinal(rank)} {name} ({val:.1f}{b['unit']})")
        return " | ".join(parts) if parts else "n/a"

    md.bullets([f"**{ctx['boards'][k]['title']}** ({ctx['boards'][k]['desc']}): {winners(k)}" for k in ctx["boards"]])

    md.h(2, "Notes on method")
    md.bullets([
        "AI acceptance uses exact boundaries because the pre-annotations were presented as spans only.",
        "Agreement figures use the final consensus dataset, which includes annotators' own votes.",
        "Time per sentence is Label Studio's lead_time. The median uses all sentences; sentences far above your own typical pace (Q3 + 3 x IQR, at least 300 s) are treated as interrupted and left out of totals.",
        "Specialty labels are those assigned to your expert group; pooled team values include all annotators.",
    ])

    items = "".join(f'<li><strong>{html.escape(label)}:</strong> <a href="{url}">{url}</a></li>' for label, url in LINKS)
    md.raw(f'<div class="links"><h2>Resources and links</h2><ul>{items}</ul></div>')
    return md.text()


CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:920px;margin:2rem auto;padding:0 1.2rem;line-height:1.55;color:#222}
h1{border-bottom:3px solid #DE4968;padding-bottom:.3rem}
h2{margin-top:2.2rem;border-bottom:1px solid #ddd;padding-bottom:.2rem;page-break-after:avoid}
h3{page-break-after:avoid}
img{max-width:100%;display:block;margin:1rem auto;page-break-inside:avoid}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem;page-break-inside:avoid}
th,td{border:1px solid #ddd;padding:.35rem .6rem;text-align:left;vertical-align:top}
th{background:#f6f1f3}
blockquote{border-left:4px solid #DE4968;margin:1rem 0;padding:.2rem 1rem;background:#faf5f6}
.example{border:1px solid #ddd;border-radius:6px;padding:.6rem 1rem;background:#fcfcfc;line-height:2.2}
.ex-title{font-weight:bold;margin:.3rem 0 0;font-size:.85rem;text-transform:uppercase;color:#777}
.links{page-break-inside:avoid}
.tree{font-family:"DejaVu Sans Mono",Menlo,monospace;font-size:.72rem;line-height:1.45;background:#f6f6f6;border:1px solid #ddd;border-radius:6px;padding:.6rem 1rem;white-space:pre-wrap}
.ent{padding:1px 4px;border-radius:3px}.ent sub{font-size:.6rem;margin-left:3px;font-weight:bold;text-transform:uppercase}
@page{size:A4;margin:16mm}
"""


def md_to_html(md_text, base_dir, title):
    import markdown
    body = markdown.markdown(md_text, extensions=["tables", "sane_lists"])

    def inline(m):
        p = os.path.join(base_dir, m.group(1))
        if os.path.exists(p):
            with open(p, "rb") as f:
                return 'src="data:image/png;base64,' + base64.b64encode(f.read()).decode() + '"'
        return m.group(0)

    body = re.sub(r'src="([^"]+\.png)"', inline, body)
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")


# ============================================================
# 7. ORCHESTRATION
# ============================================================

def make_charts(ann_id, A, analyses, metrics, team, peer_map, out_dir, min_support):
    ch = {}
    style_all = [a["style"] for a in analyses.values()]
    team_style = {k: float(np.mean([s[k] for s in style_all])) for k in A["style"]}
    plot_radar_chart(A["style"], team_style, os.path.join(out_dir, "radar.png")); ch["radar"] = "radar.png"
    plot_peer_rankings(ann_id, metrics, peer_map, os.path.join(out_dir, "peers.png")); ch["peers"] = "peers.png"
    g, ai = A["gold"], A["ai"]
    if g:
        if plot_gold_by_label(label_table(g["counts"], min_support), A["expert_labels"], os.path.join(out_dir, "gold_label.png")):
            ch["gold_label"] = "gold_label.png"
        if plot_error_taxonomy([("You", g["tax"]), ("Team (pooled)", team["tax"])], os.path.join(out_dir, "taxonomy.png")):
            ch["taxonomy"] = "taxonomy.png"
        if plot_confusions(g["conf"], os.path.join(out_dir, "confusions.png")):
            ch["confusions"] = "confusions.png"
    panels = []
    if ai:
        panels.append(("vs AI", ai["shifts"], team["shifts_ai"]))
    if g:
        panels.append(("vs final", g["shifts"], team["shifts_gold"]))
    if plot_boundary_shifts(panels, os.path.join(out_dir, "shifts.png")):
        ch["shifts"] = "shifts.png"
    if A["survival"] and plot_survivors(A["survival"]["surv_by_label"], os.path.join(out_dir, "survivors.png")):
        ch["survivors"] = "survivors.png"
    if A["timing"]:
        others = [a["timing"]["bins"] for k, a in analyses.items() if k != ann_id and a["timing"]]
        plot_fatigue(A["timing"]["bins"], others, os.path.join(out_dir, "fatigue.png")); ch["fatigue"] = "fatigue.png"
    others = [a["label_rates"] for k, a in analyses.items() if k != ann_id]
    if others:
        labs = set(A["label_rates"]).union(*[set(o) for o in others])
        diff = {l: A["label_rates"].get(l, 0) - float(np.mean([o.get(l, 0) for o in others])) for l in labs}
        if plot_label_usage(diff, A["expert_labels"], os.path.join(out_dir, "usage.png")):
            ch["usage"] = "usage.png"
    return ch


def write_team_outputs(analyses, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for k, a in analyses.items():
        r = {"annotator": k, "n_docs": a["n_docs"], **metrics[k]}
        if a["ai"]:
            r.update({"ai_total": a["ai"]["total"], "ai_added": a["ai"]["added"]})
        if a["gold"]:
            r.update({"gold_spans": a["gold"]["n_gold"],
                      **{f"tax_{t}": a["gold"]["tax"].get(t, 0) for t in TAX_ORDER}})
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "team_summary.csv"), index=False)

    lab_rows = []
    for k, a in analyses.items():
        if not a["gold"]:
            continue
        for lab, c in a["gold"]["counts"].items():
            if c["gold"]:
                sc = gold_scores({lab: c})
                lab_rows.append({"annotator": k, "label": lab, "n_gold": c["gold"],
                                 "strict_f1": sc["strict"][2], "lenient_f1": sc["lenient"][2]})
    if lab_rows:
        df = pd.DataFrame(lab_rows)
        df.to_csv(os.path.join(out_dir, "team_label_f1.csv"), index=False)
        piv = df.pivot(index="annotator", columns="label", values="strict_f1") * 100
        plt.figure(figsize=(max(10, 0.6 * piv.shape[1]), 0.5 * piv.shape[0] + 2))
        sns.heatmap(piv, annot=True, fmt=".0f", annot_kws={"size": 7}, cmap="RdPu", vmin=0, vmax=100,
                    cbar_kws={"label": "Strict F1 (%)"}, linewidths=0.4, linecolor="white")
        plt.xlabel(""); plt.ylabel("Annotator")
        _save(os.path.join(out_dir, "team_label_f1_heatmap.png"))

    ai_rows = [(k, m["acceptance"], m["deletion"], analyses[k]["ai"]["total"]) for k, m in metrics.items()
               if "acceptance" in m and "deletion" in m]
    if ai_rows:
        plt.figure(figsize=(7, 5.5))
        plt.scatter([r[1] for r in ai_rows], [r[2] for r in ai_rows], s=[r[3] / 4 + 30 for r in ai_rows], color=YOU_COLOR, alpha=0.7)
        for k, x, y, _ in ai_rows:
            plt.annotate(str(k), (x, y), xytext=(5, 5), textcoords="offset points")
        plt.xlabel("AI suggestions kept as proposed (%)")
        plt.ylabel("AI suggestions rejected (%)")
        plt.grid(linestyle="--", alpha=0.5)
        _save(os.path.join(out_dir, "team_ai_behaviour.png"))

    tax_rows = [(f"Ann {k}", a["gold"]["tax"]) for k, a in analyses.items() if a["gold"]]
    plot_error_taxonomy(tax_rows, os.path.join(out_dir, "team_error_taxonomy.png"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotator-dir", default=ANNOTATOR_DIR)
    p.add_argument("--referent-dir", default=REFERENT_DIR)
    p.add_argument("--gold-path", default=GOLD_PATH)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--only", type=int, nargs="*", help="only render these annotator IDs")
    p.add_argument("--no-html", action="store_true", help="skip HTML output")
    p.add_argument("--pdf", action="store_true", help="also write PDF (needs weasyprint)")
    p.add_argument("--package", action="store_true",
                   help="build one gift folder per annotator (PDF report, game, paper, raw data, game data); implies PDF")
    p.add_argument("--packages-dir", default=PACKAGES_DIR)
    p.add_argument("--min-label-support", type=int, default=MIN_LABEL_SUPPORT)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("=== Loading data ===")
    all_docs, all_timing = load_all_annotators(args.annotator_dir)
    ref = load_reference_data(args.referent_dir)
    gold = load_gold(args.gold_path)
    expert_map = get_expert_mapping()
    print(f"Annotators: {sorted(all_docs)} | reference docs: {len(ref)} | gold docs: {len(gold)}")

    print("=== Computing metrics ===")
    analyses = {a: analyze_annotator(a, d, ref, gold, all_timing.get(a, {}), expert_map.get(a, set()))
                for a, d in all_docs.items()}
    metrics = {a: flatten_metrics(A, MACE_SCORES) for a, A in analyses.items()}
    boards = generate_leaderboards(metrics)

    team = {"tax": Counter(), "shifts_ai": [], "shifts_gold": []}
    for A in analyses.values():
        if A["gold"]:
            team["tax"].update(A["gold"]["tax"])
            team["shifts_gold"] += A["gold"]["shifts"]
        if A["ai"]:
            team["shifts_ai"] += A["ai"]["shifts"]
    team_ai = {k: mean_of([m.get(k) for m in metrics.values()]) for k in ("acceptance", "modification", "deletion")}

    thanks = load_thank_you_notes()
    if args.package:
        args.pdf = True

    os.makedirs(args.output_dir, exist_ok=True)
    for ann_id, A in analyses.items():
        if args.only and ann_id not in args.only:
            continue
        print(f"--- Report for annotator {ann_id} ---")
        name, _ = thanks.get(ann_id, (None, None))
        tmp = tempfile.TemporaryDirectory() if args.package else None   # package mode keeps only the PDF
        out_dir = tmp.name if tmp else os.path.join(args.output_dir, f"ann_{ann_id}")
        os.makedirs(out_dir, exist_ok=True)
        others = [k for k in analyses if k != ann_id]
        rng = random.Random(ann_id * 42)       # deterministic per annotator, same as before
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        rng.shuffle(letters)
        peer_map = {oid: f"Annotator {letters[i]}" for i, oid in enumerate(others)}

        charts = make_charts(ann_id, A, analyses, metrics, team, peer_map, out_dir, args.min_label_support)
        ex = None
        if A["gold"] and A["gold"]["contested"]:
            ex = example_html(A["gold"]["contested"], all_docs[ann_id], gold)
        ctx = {"metrics": metrics, "charts": charts, "peer_map": peer_map, "boards": boards,
               "takeaways": build_takeaways(ann_id, metrics), "team_ai": team_ai,
               "team_shifts_ai": team["shifts_ai"], "team_shifts_gold": team["shifts_gold"],
               "min_support": args.min_label_support, "example": ex, "thanks": thanks.get(ann_id),
               "package": package_layout(ann_id, name, args.annotator_dir) if args.package else None}
        report = render_report(A, ctx)
        md_path = os.path.join(out_dir, "report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report)
        if not args.no_html or args.pdf:
            try:
                html_text = md_to_html(report, out_dir, f"CliReNER annotation report, annotator {ann_id}")
                with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
                    f.write(html_text)
                if args.pdf:
                    from weasyprint import HTML
                    HTML(string=html_text, base_url=out_dir).write_pdf(os.path.join(out_dir, "report.pdf"))
            except ImportError as e:
                print(f"  Skipped HTML/PDF ({e}). pip install markdown weasyprint")
        if args.package:
            pdf_path = os.path.join(out_dir, "report.pdf")
            if os.path.exists(pdf_path):
                pkg = build_package(ann_id, name, pdf_path, ctx["package"], args.annotator_dir, args.packages_dir)
                print(f"  Package: {pkg}")
            else:
                print("  ERROR: PDF was not created; package skipped.")
            tmp.cleanup()
        else:
            print(f"  Saved reports in: {out_dir}")

    if not args.package:
        print("=== Team outputs ===")
        write_team_outputs(analyses, metrics, os.path.join(args.output_dir, "TEAM"))
        print(f"All reports generated in: {args.output_dir}")


if __name__ == "__main__":
    main()