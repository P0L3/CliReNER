from dataset_processing import (
    cwed4eta_process_json_file,
    convert_to_token_spans,
    CLIRENER_LABELS_V1,
    process_directory_of_json_files,
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

from dataset_processing import hf_dataset_to_gliner_format
from EXPERIMENTS.evaluate_gold import load_and_merge_gold_data

# Ensure target directories exist locally
os.makedirs("PLOTS", exist_ok=True)
os.makedirs("RESULTS", exist_ok=True)

# --- Loading the Reference Datasets ---

# 1. Referent data used for CliReNER GOLD pre-annotation (silver-fine-tuned GLiNER,
#    spans only, shown to the 12 expert annotators)
rmd_dir_gold = "RESULTS/FT_GOLD/"
data_referent_gold = convert_to_token_spans(process_directory_of_json_files(rmd_dir_gold, -1))
data_referent_gold = list({d['id']: d for d in data_referent_gold}.values())

# 2. Referent data used during author CliReNER SILVER pre-annotation (off-the-shelf
#    GLiNER, spans + types, shown only to the author)
rmd_dir_silver = "RESULTS/gliner_noft/"
data_referent_silver = convert_to_token_spans(process_directory_of_json_files(rmd_dir_silver, -1))
data_referent_silver = list({d['id']: d for d in data_referent_silver}.values())


# --- Loading the Individual Coder Datasets ---

ANNOTATOR_DIR = "/home/p0l3/RAD/DROP/CLIRENER/ANNOTATORS/"
data_0 = convert_to_token_spans(cwed4eta_process_json_file(ANNOTATOR_DIR + "0/" + "OG.json"))
data_1 = convert_to_token_spans(cwed4eta_process_json_file(ANNOTATOR_DIR + "1/" + "G3_10226.json", [4, 1, 5]))
data_2 = convert_to_token_spans(cwed4eta_process_json_file(ANNOTATOR_DIR + "2/" + "G4_5326.json", [1, 6]))
data_3 = convert_to_token_spans(cwed4eta_process_json_file(ANNOTATOR_DIR + "3/" + "G4_4326.json", [1, 7]))
data_4 = convert_to_token_spans(cwed4eta_process_json_file(ANNOTATOR_DIR + "4/" + "G2_5326_1.json", [1, 8]))
data_5 = convert_to_token_spans(process_directory_of_json_files(ANNOTATOR_DIR + "5/", [1, 9]))
data_6 = convert_to_token_spans(cwed4eta_process_json_file(ANNOTATOR_DIR + "6/" + "G1_15126.json", [13]))
data_7 = convert_to_token_spans(cwed4eta_process_json_file(ANNOTATOR_DIR + "7/" + "G6.json", [1, 9]))
data_8 = convert_to_token_spans(cwed4eta_process_json_file(ANNOTATOR_DIR + "8/" + "G6_5326.json", [12]))
data_9 = convert_to_token_spans(cwed4eta_process_json_file(ANNOTATOR_DIR + "9/" + "G2_5326.json", [1, 11]))
data_10 = convert_to_token_spans(cwed4eta_process_json_file(ANNOTATOR_DIR + "10/" + "G3_10226_2.json", [1, 15]))
data_11 = convert_to_token_spans(cwed4eta_process_json_file(ANNOTATOR_DIR + "11/" + "G5_21126.json", [4, 1, 14]))

# Corrected loading logic for data_12
with open(ANNOTATOR_DIR + "12/" + "G5_4326.json", "r") as f:
    data_12 = json.load(f)


print("Loading and merging Gold dataset from HF...")
# dataset_hf = load_and_merge_gold_data("P0L3/CliReNER_v_1_1_28_GOLD")
# data_gold_merged = hf_dataset_to_gliner_format(dataset_hf[0]["test"], dataset_hf[1])
data_gold_merged = convert_to_token_spans(cwed4eta_process_json_file(ANNOTATOR_DIR + "CONSENSUS/" + "6326.json", [1]))


def _unpack(span):
    if isinstance(span, dict):
        return span['start'], span['end'], span['label']
    return span[0], span[1], span[2]


def _new_stats():
    return {
        "total_model_spans": 0,
        "exact_matches": 0,
        "exact_matches_with_type": 0,
        "modified_boundaries": 0,
        "deleted_by_annotator": 0,
        "added_by_annotator": 0,
    }



def _compute_rates(s):
    total = s["total_model_spans"]

    # These three partition the reference spans exactly once each:
    # every suggested span is either accepted as-is, boundary-modified,
    # or deleted outright. All three now share the same denominator
    # (total_model_spans) so they sum to ~1.0 by construction.
    s["acceptance_rate"] = s["exact_matches"] / total if total else float("nan")
    s["boundary_mod_rate"] = s["modified_boundaries"] / total if total else float("nan")
    s["deletion_rate"] = s["deleted_by_annotator"] / total if total else float("nan")

    # Secondary, non-partition metrics: strict_acceptance_rate is a SUBSET
    # of acceptance_rate (exact boundary AND correct type, not an
    # independent bucket), and addition_rate is normalized against
    # total_model_spans just for scale but describes spans with no
    # corresponding reference span at all -- it can exceed 100%.
    s["strict_acceptance_rate"] = (
        s["exact_matches_with_type"] / total if total else float("nan")
    )
    s["addition_rate"] = (
        s["added_by_annotator"] / total if total else float("nan")
    )

    # Sanity check: catches any future regression that breaks the partition.
    if total:
        partition_sum = s["acceptance_rate"] + s["boundary_mod_rate"] + s["deletion_rate"]
        assert abs(partition_sum - 1.0) < 1e-6, (
            f"Partition check failed: acceptance+modification+deletion = "
            f"{partition_sum:.6f}, expected ~1.0"
        )

    return s


def calculate_span_modification_rate(data_n, data_referent, by_type=False):
    """
    Calculates how often an annotator modified boundaries/labels of model
    suggestions, relative to a SINGLE, appropriate reference set.
    """
    dict_ann = {d['id']: d for d in data_n}
    dict_ref = {d['id']: d for d in data_referent}
    common_ids = set(dict_ann.keys()) & set(dict_ref.keys())
    if not common_ids:
        return None

    overall = _new_stats()
    by_type_stats = {}

    for doc_id in common_ids:
        ann_spans = dict_ann[doc_id]['ner']
        ref_spans = dict_ref[doc_id]['ner']

        overall["total_model_spans"] += len(ref_spans)
        matched_ann_indices = set()

        for r_span in ref_spans:
            r_start, r_end, r_label = _unpack(r_span)
            type_bucket = by_type_stats.setdefault(r_label, _new_stats())
            type_bucket["total_model_spans"] += 1

            candidates = []
            for i, a_span in enumerate(ann_spans):
                a_start, a_end, a_label = _unpack(a_span)
                # Inclusive index overlap check
                if r_start <= a_end and a_start <= r_end:
                    candidates.append((i, a_start, a_end, a_label))

            if not candidates:
                overall["deleted_by_annotator"] += 1
                type_bucket["deleted_by_annotator"] += 1
                continue

            exact = [c for c in candidates if c[1] == r_start and c[2] == r_end]
            chosen = exact[0] if exact else max(
                candidates, key=lambda c: min(c[2], r_end) - max(c[1], r_start)
            )
            i, a_start, a_end, a_label = chosen
            matched_ann_indices.add(i)

            if a_start == r_start and a_end == r_end:
                overall["exact_matches"] += 1
                type_bucket["exact_matches"] += 1
                if a_label == r_label:
                    overall["exact_matches_with_type"] += 1
                    type_bucket["exact_matches_with_type"] += 1
            else:
                overall["modified_boundaries"] += 1
                type_bucket["modified_boundaries"] += 1

        overall["added_by_annotator"] += len(ann_spans) - len(matched_ann_indices)

    overall = _compute_rates(overall)
    if by_type:
        by_type_stats = {k: _compute_rates(v) for k, v in by_type_stats.items()}
        return overall, by_type_stats
    return overall


# --- Data Collection: each annotator vs. ONLY the reference they actually saw ---
records = []
by_type_pool = {}

author_stats = calculate_span_modification_rate(data_0, data_referent_silver)
if author_stats:
    row = author_stats.copy()
    row['Annotator'] = 'Author'
    row['Reference'] = 'Silver Reference'
    records.append(row)
else:
    print("WARNING: no ID overlap between Author annotations and Silver reference.")

expert_annotators = {
    'Ann_1': data_1, 'Ann_2': data_2, 'Ann_3': data_3, 'Ann_4': data_4,
    'Ann_5': data_5, 'Ann_6': data_6, 'Ann_7': data_7, 'Ann_8': data_8,
    'Ann_9': data_9, 'Ann_10': data_10, 'Ann_11': data_11, 'Ann_12': data_12,
}

for label, ann_data in expert_annotators.items():
    result = calculate_span_modification_rate(ann_data, data_referent_gold, by_type=True)
    if result is None:
        print(f"WARNING: no ID overlap between {label} and Gold reference.")
        continue
    overall_stats, type_stats = result
    row = overall_stats.copy()
    row['Annotator'] = label
    row['Reference'] = 'Gold Reference'
    records.append(row)

    for etype, s in type_stats.items():
        pool = by_type_pool.setdefault(etype, _new_stats())
        for k in pool:
            pool[k] += s[k]

gold_merged_stats = calculate_span_modification_rate(data_gold_merged, data_referent_gold)
if gold_merged_stats:
    row_merged = gold_merged_stats.copy()
    row_merged['Annotator'] = 'Gold Merged'
    row_merged['Reference'] = 'Gold Reference'
    records.append(row_merged)
else:
    print("WARNING: no ID overlap between Gold Merged dataset and Gold Reference -- check alignment.")

df = pd.DataFrame(records)


# --- Apply Global Design Parameters ---
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


# =========================================================
# Combined dense plot 1: one heatmap, annotators (rows) x metrics (columns).
# =========================================================
metric_cols = ['acceptance_rate', 'strict_acceptance_rate', 'boundary_mod_rate',
                'deletion_rate', 'addition_rate']
metric_labels = ['Acceptance\n(boundary)', 'Acceptance\n(boundary+type)',
                  'Boundary\nmodification', 'Deletion', 'Addition']

plot_df = df.set_index('Annotator')[metric_cols].copy()
plot_df.columns = metric_labels
n_counts = df.set_index('Annotator')['total_model_spans']

fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(plot_df))))
sns.heatmap(
    plot_df * 100,
    annot=True, fmt=".1f", cmap="RdPu",
    cbar_kws={'label': 'Rate (%)'},
    linewidths=0.5, linecolor='white',
    ax=ax,
)
# ax.set_title(
#     'Annotator Boundary/Type Editing Behaviour vs. Pre-Annotation\n'
#     '(Author row: Silver Reference \u00b7 Expert rows: Gold Reference)',
#     fontsize=13, pad=12,
# )
ax.set_ylabel('')
ax.set_xlabel('')
# for i, (annotator, n) in enumerate(n_counts.items()):
#     ax.text(len(metric_cols) + 0.15, i + 0.5, f"N={int(n)}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig("PLOTS/boundary_editing_combined.png", dpi=300, bbox_inches='tight')
plt.show()


# =========================================================
# Combined dense plot 2: per-entity-type editing rates (pooled experts, Gold Ref)
# =========================================================
type_rows = []
for etype, counts in by_type_pool.items():
    rates = _compute_rates(counts.copy())
    rates['EntityType'] = etype
    type_rows.append(rates)

type_df = pd.DataFrame(type_rows).set_index('EntityType')
type_df = type_df.sort_values('acceptance_rate', ascending=False)

fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(type_df))))

# Metric columns and formatted labels for type-level plotting
type_cols_to_plot = ['acceptance_rate', 'strict_acceptance_rate', 'boundary_mod_rate', 'deletion_rate']
type_metric_labels = ['Acceptance\n(boundary)', 'Acceptance\n(boundary+type)',
                       'Boundary\nmodification', 'Deletion']

plot_type_df = type_df[type_cols_to_plot].copy()
plot_type_df.columns = type_metric_labels
n_type_counts = type_df['total_model_spans']

sns.heatmap(
    plot_type_df * 100,
    annot=True, fmt=".1f", cmap="RdPu",
    cbar_kws={'label': 'Rate (%)'},
    linewidths=0.5, linecolor='white',
    ax=ax,
)
# ax.set_title(
#     'Pre-Annotation Editing Rate by Entity Type\n'
#     '(pooled across 12 expert annotators, Gold Reference)',
#     fontsize=13, pad=12,
# )
ax.set_ylabel('')

# # Plot N counts on the right margin of the second plot
# for i, (etype, n) in enumerate(n_type_counts.items()):
#     ax.text(len(type_cols_to_plot) + 0.15, i + 0.5, f"N={int(n)}", va='center', fontsize=8)

plt.tight_layout()

plt.savefig("PLOTS/boundary_editing_by_type.png", dpi=300, bbox_inches='tight')
plt.show()

plot_df.to_csv("RESULTS/boundary_edit_rate.csv")
plot_type_df.to_csv("RESULTS/boundary_edit_rate_per_type.csv")

"""
Below are the updated formal descriptions of the five metrics, revised to reflect the unified denominator ($M_{\text{total}}$) for your paper’s **Methodology** or **Evaluation Metrics** section.

$$M_{\text{total}} = \text{Total pre-annotated suggestions generated by the reference model}$$
$$S_{\text{exact}} = \text{Suggested spans with exact matching boundaries (regardless of type)}$$
$$S_{\text{strict}} = \text{Suggested spans with exact matching boundaries and matching entity types}$$
$$S_{\text{modified}} = \text{Suggested spans overlapping with human spans, but requiring resized boundaries}$$
$$S_{\text{deleted}} = \text{Suggested spans completely removed/rejected by the annotator}$$
$$S_{\text{added}} = \text{Newly created spans introduced by the annotator}$$

---

#### I. Acceptance Rate (Boundary)
$$\text{Acceptance Rate (Boundary)} = \frac{S_{\text{exact}}}{M_{\text{total}}}$$
*   **Definition:** Measures the proportion of model-generated entity suggestions where the human annotator accepted the proposed start and end token boundaries exactly [3], irrespective of the predicted entity type.
*   **Scientific Context:** This metric isolates the locational performance of the pre-annotation model. It quantifies the system's capacity to precisely delineate raw entity segments without being penalized for downstream class label mistakes.

#### II. Acceptance Rate (Boundary + Type) / Strict Acceptance Rate
$$\text{Strict Acceptance Rate} = \frac{S_{\text{strict}}}{M_{\text{total}}}$$
*   **Definition:** Calculates the proportion of pre-annotated spans where the annotator accepted both the proposed boundaries and the predicted entity type exactly as suggested [3], making zero adjustments.
*   **Scientific Context:** This represents the strictest form of pre-annotation utility. It measures the frequency of "zero-correction" events—instances where the machine output required zero cognitive or manual editing from the human annotator. This metric is a mathematical subset of the standard Acceptance Rate.

#### III. Boundary Modification Rate (BMR)
$$\text{Boundary Modification Rate} = \frac{S_{\text{modified}}}{M_{\text{total}}}$$
*   **Definition:** Evaluates the proportion of all model-generated suggestions where the human annotator chose to retain the entity but modified its token boundaries [3] (by shrinking, expanding, or shifting them).
*   **Scientific Context:** Unlike conditional boundary rates, this global metric directly quantifies the fraction of total model suggestions that, while locational-accurate enough to avoid deletion, still demanded manual boundary adjustment. It exposes the precise fraction of machine-suggested boundaries that introduced manual editing overhead.

#### IV. Deletion Rate
$$\text{Deletion Rate} = \frac{S_{\text{deleted}}}{M_{\text{total}}}$$
*   **Definition:** The proportion of suggested spans that the human annotator completely rejected and deleted from the text, indicating no overlapping annotations exist for that segment [3].
*   **Scientific Context:** This metric serves as a direct proxy for the pre-annotation system's false-positive rate. It measures the cognitive overhead introduced by the model generating hallucinated or contextually irrelevant entities that the human must actively reject.

#### V. Addition Rate
$$\text{Addition Rate} = \frac{S_{\text{added}}}{M_{\text{total}}}$$
*   **Definition:** The volume of entirely new entity spans manually introduced by the annotator (which did not overlap with any model suggestions), normalized by the total number of suggestions generated [3].
*   **Scientific Context:** The Addition Rate highlights the omissions and false-negative trends of the pre-annotation model (recall gaps) [3]. A high Addition Rate shows where the pre-annotation model failed to capture target entities, forcing human annotators to find and tag them from scratch.

"""


"""
MATHEMATICAL INTERACTION BETWEEN THE METRICS:

By utilizing the total pre-annotated spans (total_model_spans) as the unified 
denominator, the metrics are governed by an exact mathematical partition, 
a clean subset relationship, and an independent scale factor.

1. The Tripartite Partition of Model Suggestions (Sums to 100%)
   Every pre-annotation span suggested by the model (total_model_spans) 
   must fall into exactly one of three mutually exclusive outcomes:
     - Accepted with exact boundaries (exact_matches)
     - Retained, but boundaries were modified (modified_boundaries)
     - Rejected/deleted completely (deleted_by_annotator)
   
   Because total_model_spans = exact_matches + modified_boundaries + deleted_by_annotator,
   dividing this partition by total_model_spans guarantees:
   
     Acceptance Rate + Boundary Modification Rate + Deletion Rate = 100%
     
   This identity is strictly validated by the assertions in this code block.

2. Subset Relationship (Strict Acceptance)
   The Strict Acceptance Rate (strict_acceptance_rate) is not an independent 
   outcome bucket, but rather a formal subset of the baseline Acceptance Rate. 
   It represents the proportion of exact boundary matches that also achieved 
   perfect class label agreement:
   
     strict_acceptance_rate <= acceptance_rate

3. Addition Rate Independence
   The Addition Rate measures human-initiated annotations (added_by_annotator)
   that have no counterpart in the pre-annotation output, normalized by the 
   total suggested spans for scaling. Because it is independent of the model's 
   suggestion outcomes, it is mathematically unbounded and can exceed 100% if 
   the annotator introduces more new spans than the model originally suggested.
"""