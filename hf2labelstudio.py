"""
Converts a CliReNER HF NER dataset (e.g. SILVER) into a Label Studio import
file for Relation Extraction annotation, pre-loaded with the original NER
spans as predictions, with each task traceable back to its source paper via
per-row paper_id/sentence_id (via PROVENANCE/hf_to_original_map.json),
rather than one shared paper_id for the whole export.
"""

def get_hf_token_char_spans(text, tokens):
    """
    Recomputes character-level spans for each token, assuming `tokens` came
    from dataset_processing.tokenize_text(text) — true for every CliReNER
    SILVER/GOLD HF dataset. Mirrors the exact same left-to-right `text.find`
    logic used when the token-level dataset was originally built in
    `transform_to_ner_format`, so spans line up exactly.
    """
    spans = []
    current_pos = 0
    for token in tokens:
        start = text.find(token, current_pos)
        if start == -1:
            # Shouldn't happen with CliReNER's own tokenizer, but don't crash.
            start = current_pos
        end = start + len(token)
        spans.append((start, end))
        current_pos = end
    return spans


def hf_row_entities_char_level(row, label_names):
    """
    Given one row of a CliReNER HF dataset (with 'text' or 'tokens', plus
    'ner_tags'), returns entities as char-offset dicts:
        [{'start': int, 'end': int, 'text': str, 'label': str}, ...]
    ready for Label Studio's 'value' block. Unlike hf_dataset_to_gliner_format,
    this slices directly out of the row's own text so 'text' always matches
    text[start:end] exactly (required for LS to render highlights correctly).
    """
    text = row.get("text") or " ".join(row["tokens"])
    tokens = row["tokens"]
    tags = row["ner_tags"]
    token_spans = get_hf_token_char_spans(text, tokens)

    entities = []
    current_start_tok = None
    current_label = None

    def _flush(end_tok_exclusive):
        nonlocal current_start_tok, current_label
        if current_start_tok is None:
            return
        char_start = token_spans[current_start_tok][0]
        char_end = token_spans[end_tok_exclusive - 1][1]
        entities.append({
            "start": char_start,
            "end": char_end,
            "text": text[char_start:char_end],
            "label": current_label,
        })
        current_start_tok = None
        current_label = None

    for i, tag_id in enumerate(tags):
        tag_name = label_names[tag_id]
        if tag_name.startswith("B-"):
            _flush(i)
            current_start_tok = i
            current_label = tag_name[2:]
        elif tag_name.startswith("I-"):
            clean = tag_name[2:]
            if current_start_tok is None:
                current_start_tok = i
                current_label = clean
            elif current_label != clean:
                _flush(i)
                current_start_tok = i
                current_label = clean
        else:  # "O"
            _flush(i)

    _flush(len(tags))
    return entities


def load_hf_provenance_lookup(provenance_path, hf_dataset_name):
    """
    Loads hf_to_original_map.json (from build_hf_provenance_map.py) and builds
    a dict keyed by each row's own HF 'id', scoped to one --hf-dataset target
    (one provenance file can cover several).

    Value per hf_id:
        {
            "status": "unique" | "ambiguous",
            "paper_id": str or None,
            "sentence_id": str/int or None,
            "compound_id": str or None,
            "candidates": [...],   # kept for ambiguous rows, for manual review
        }
    """
    try:
        with open(provenance_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Provenance file not found at '{provenance_path}'. "
              f"Falling back to hf_id-only resolution for every row.")
        return {}

    lookup = {}
    for rec in records:
        if rec.get("hf_dataset") != hf_dataset_name:
            continue
        hf_id = rec.get("current_hf_id")
        if not hf_id:
            continue

        status = rec.get("status", "unmatched")
        candidates = rec.get("candidates", [])
        entry = {"status": status, "paper_id": None, "sentence_id": None,
                  "compound_id": None, "candidates": candidates}

        if status in ("unique", "ambiguous") and candidates:
            c = candidates[0]  # deterministic pick; ambiguous ones stay flagged
            entry["paper_id"] = c.get("paper_id")
            entry["sentence_id"] = c.get("sentence_id")
            entry["compound_id"] = c.get("compound_id")

        lookup[hf_id] = entry

    return lookup


def load_paper_metadata(paper_metadata_path):
    """Loads paper_metadata.json (from paper_metadata_lookup.py)."""
    try:
        with open(paper_metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Paper metadata file not found at '{paper_metadata_path}'. "
              f"Continuing without title/author/DOI enrichment.")
        return {}


def convert_to_labelstudio_format_v2(
    hf_rows,
    label_names,
    provenance_lookup=None,
    paper_metadata=None,
    model_version="P0L3/CliReNER_v_1_1_28_SILVER",
    id_fallback="unknown",
):
    """
    Per-row sibling of convert_to_labelstudio_format(), for HF NER datasets
    that need to become RE-ready Label Studio imports. Where the legacy
    function takes a list of char-offset 'entities' dicts plus one shared
    paper_id for the whole batch, this one takes raw HF rows ('id', 'text',
    'tokens', 'ner_tags') and resolves paper_id/sentence_id independently for
    EACH row via `provenance_lookup` (see load_hf_provenance_lookup), with a
    fallback to parsing the row's own 'id' if it already looks like a real
    "paper_id-sentence_id" compound key.

    NER spans are converted to char-offset Label Studio 'labels' predictions
    (pre-annotations for the annotator to draw relations on top of) — no
    'relations' are pre-populated, since that's the actual annotation task.

    Returns:
        (labelstudio_tasks, qc_report)
        qc_report: counts per resolution status, e.g.
            {"unique": 4200, "ambiguous": 12, "hf_id_fallback": 30, "no_provenance": 5}
    """
    provenance_lookup = provenance_lookup or {}
    paper_metadata = paper_metadata or {}

    labelstudio_data = []
    qc_report = Counter()

    for row in hf_rows:
        hf_id = row.get("id", "")
        text = row.get("text") or " ".join(row["tokens"])
        entities = hf_row_entities_char_level(row, label_names)

        prov = provenance_lookup.get(hf_id)
        if prov is not None:
            status = prov["status"]
            paper_id = prov["paper_id"]
            sentence_id = prov["sentence_id"]
            compound_id = prov["compound_id"] or hf_id
        elif hf_id and "-" in hf_id:
            # Newer pushes may already carry a real compound id in 'id' itself.
            paper_id, sentence_id = hf_id.rsplit("-", 1)
            status = "hf_id_fallback"
            compound_id = hf_id
        else:
            paper_id, sentence_id = None, None
            status = "no_provenance"
            compound_id = hf_id
        qc_report[status] += 1

        meta = paper_metadata.get(str(paper_id), {}) if paper_id is not None else {}

        data_payload = {
            "sentence": text,
            "compound_id": compound_id,
            "hf_id": hf_id,
            "paper_id": paper_id if paper_id is not None else id_fallback,
            "sentence_id": sentence_id if sentence_id is not None else id_fallback,
            "provenance_status": status,
            "title": meta.get("Title"),
            "authors": meta.get("Authors"),
            "doi": meta.get("DOI"),
            "journal": meta.get("Journal"),
            "date": meta.get("Date"),
        }

        predictions = {
            "model_version": model_version,
            "score": 0.5,
            "result": [
                {
                    "id": str(uuid.uuid4())[:8],
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": ent["start"],
                        "end": ent["end"],
                        "text": ent["text"],
                        "labels": [ent["label"]],
                    },
                }
                for ent in entities
            ],
        }

        labelstudio_data.append({"data": data_payload, "predictions": [predictions]})

    return labelstudio_data, dict(qc_report)



import json
import argparse

from datasets import load_dataset, concatenate_datasets
from collections import Counter
import uuid

# from dataset_processing import (
#     convert_to_labelstudio_format_v2,
#     load_hf_provenance_lookup,
#     load_paper_metadata,
# )


def main():
    parser = argparse.ArgumentParser(description="Convert a CliReNER HF dataset to Label Studio RE-ready tasks.")
    parser.add_argument("--hf_name", type=str, default="P0L3/CliReNER_v_1_1_28_SILVER")
    parser.add_argument("--provenance", type=str, default="PROVENANCE/hf_to_original_map.json")
    parser.add_argument("--paper_metadata", type=str, default="PROVENANCE/paper_metadata.json")
    parser.add_argument("--output", type=str, default=None,
                         help="Defaults to <shortened_hf_name>_labelstudio_re.json")
    args = parser.parse_args()

    output_path = args.output or f"RESULTS/{args.hf_name.split('/')[-1]}_labelstudio_re.json"

    print(f"Loading from HF: {args.hf_name} ...")
    ds = load_dataset(args.hf_name)
    merged = concatenate_datasets([ds[split] for split in ds.keys()])
    label_names = merged.features["ner_tags"].feature.names
    print(f"Loaded {len(merged)} rows across splits {list(ds.keys())}.")

    print(f"Loading provenance map from {args.provenance} ...")
    provenance_lookup = load_hf_provenance_lookup(args.provenance, args.hf_name)
    print(f"  -> resolved provenance for {len(provenance_lookup)} rows of this dataset.")

    print(f"Loading paper metadata from {args.paper_metadata} ...")
    paper_metadata = load_paper_metadata(args.paper_metadata)
    print(f"  -> {len(paper_metadata)} papers with metadata.")

    print("Converting to Label Studio format ...")
    labelstudio_tasks, qc_report = convert_to_labelstudio_format_v2(
        merged,
        label_names,
        provenance_lookup=provenance_lookup,
        paper_metadata=paper_metadata,
        model_version=args.hf_name,
    )

    print("\nProvenance QC report (per row):")
    for status, count in sorted(qc_report.items()):
        print(f"  {status:<16}: {count}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labelstudio_tasks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(labelstudio_tasks)} Label Studio tasks to: {output_path}")


if __name__ == "__main__":
    main()