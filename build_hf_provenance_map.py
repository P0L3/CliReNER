"""
build_hf_provenance_map.py

Builds a local JSON sidecar mapping HF dataset sentences back to their
original (paper_id, sentence_id) provenance, for datasets that were
already pushed to the Hub before `transform_to_ner_format` was patched
to preserve real IDs.

This does NOT modify anything on the Hub. It only reads your local
Label Studio / consensus JSON exports and the already-published HF
datasets, and writes a JSON file you can join against by exact text.

Usage:
    python build_hf_provenance_map.py \
        --source DATA/LABEL_STUDIO/project-30-at-2025-11-14-12-19-2a7464a5.json \
        --hf-dataset P0L3/CliReNER_v_1_1_28_GOLD \
        --hf-dataset P0L3/CliReNER_v_1_1_28_SILVER \
        --output PROVENANCE/hf_to_original_map.json

# --source consensus_for_labelstudio_adjustedvote_5326_v0.json \

Pass one --source per Label Studio / consensus JSON file that ever fed
into ANY of the target HF datasets, and one --hf-dataset per Hub repo
you want provenance for. You can list several of each.

Matching is done on EXACT sentence text, so it degrades gracefully:
  - unique match   -> provenance recorded
  - duplicate text in your sources -> recorded as "ambiguous", all
    candidate (paper_id, sentence_id) pairs kept so you can resolve
    manually rather than guessing
  - no match       -> recorded as "unmatched", so you know it needs a
    manual look (e.g. text was normalized/whitespace-cleaned somewhere
    between annotation and HF push)
"""

import argparse
import json
import os
from collections import defaultdict

from datasets import load_dataset, concatenate_datasets


def extract_text_id_pairs(file_path):
    """
    Pulls (text, compound_id, paper_id, sentence_id) tuples out of any
    Label Studio-shaped JSON export used across this project. Works
    regardless of which annotator/consensus method produced the file --
    it only needs 'data.sentence', 'data.paper_id', 'data.sentence_id',
    which every task/export in this pipeline carries.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    pairs = []
    skipped = 0
    for task in tasks:
        data = task.get("data", {})
        text = data.get("sentence")
        paper_id = data.get("paper_id")
        sentence_id = data.get("sentence_id")

        if text is None or paper_id is None or sentence_id is None:
            skipped += 1
            continue

        compound_id = f"{paper_id}-{sentence_id}"
        pairs.append((text, compound_id, paper_id, sentence_id))

    if skipped:
        print(f"  [{file_path}] skipped {skipped} tasks missing sentence/paper_id/sentence_id")
    return pairs


def build_source_lookup(source_paths):
    """
    text -> list of {compound_id, paper_id, sentence_id, source_file}
    Kept as a list (not a single dict) so duplicate texts across
    sources/papers are preserved rather than silently overwritten.
    """
    lookup = defaultdict(list)
    for path in source_paths:
        print(f"Reading source: {path}")
        for text, compound_id, paper_id, sentence_id in extract_text_id_pairs(path):
            entry = {
                "compound_id": compound_id,
                "paper_id": paper_id,
                "sentence_id": sentence_id,
                "source_file": path,
            }
            if entry not in lookup[text]:
                lookup[text].append(entry)
    return lookup


def get_row_text(row):
    """Some pushed dataset variants have a 'text' column, some don't -- fall back to joining tokens."""
    if "text" in row and row["text"]:
        return row["text"]
    return " ".join(row["tokens"])


def match_hf_dataset(hf_name, source_lookup):
    print(f"\nLoading HF dataset: {hf_name}")
    ds = load_dataset(hf_name)
    combined = concatenate_datasets([ds[split] for split in ds.keys()])

    results = []
    n_unique = 0
    n_ambiguous = 0
    n_unmatched = 0

    for row in combined:
        text = get_row_text(row)
        candidates = source_lookup.get(text, [])

        if len(candidates) == 0:
            status = "unmatched"
            n_unmatched += 1
        elif len(candidates) == 1:
            status = "unique"
            n_unique += 1
        else:
            status = "ambiguous"
            n_ambiguous += 1

        results.append({
            "hf_dataset": hf_name,
            "current_hf_id": row.get("id"),
            "text": text,
            "status": status,
            "candidates": candidates,
        })

    total = len(results)
    unmatched_pct = (n_unmatched / total) if total else 0.0
    print(f"  {hf_name}: {n_unique}/{total} unique matches, "
          f"{n_ambiguous} ambiguous, {n_unmatched} unmatched "
          f"({unmatched_pct:.1%} unmatched)")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", required=True,
                         help="Path to a Label Studio JSON export or consensus file. Repeatable.")
    parser.add_argument("--hf-dataset", action="append", required=True,
                         help="HF dataset repo id to map back to provenance. Repeatable.")
    parser.add_argument("--output", default="hf_provenance_map.json",
                         help="Where to write the resulting JSON sidecar.")
    args = parser.parse_args()

    source_lookup = build_source_lookup(args.source)
    print(f"\nBuilt source lookup: {len(source_lookup)} unique sentence texts "
          f"across {len(args.source)} source file(s).")

    all_results = []
    for hf_name in args.hf_dataset:
        all_results.extend(match_hf_dataset(hf_name, source_lookup))

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved provenance map to: {args.output}")

    # Quick global summary across all requested datasets
    n_ambiguous_total = sum(1 for r in all_results if r["status"] == "ambiguous")
    n_unmatched_total = sum(1 for r in all_results if r["status"] == "unmatched")
    if n_ambiguous_total or n_unmatched_total:
        print(f"NOTE: {n_ambiguous_total} ambiguous + {n_unmatched_total} unmatched rows "
              f"across all datasets need a manual look before you rely on this map.")


if __name__ == "__main__":
    main()