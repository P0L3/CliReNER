"""
build_paper_metadata_lookup.py

Given the sentence-level provenance map produced by
build_hf_provenance_map.py, pulls the corresponding paper-level
metadata (title, authors, DOI, journal, date, ...) out of the original
ED4RE_2603.pickle dataframe and writes it out as a compact JSON lookup
keyed by paper_id.

Kept SEPARATE from (rather than merged into) the sentence-level
provenance map on purpose: many sentences share the same paper, so a
per-paper file avoids repeating Title/Authors/etc. thousands of times.
Join by paper_id whenever you need it, e.g.:

    with open("PROVENANCE/paper_metadata.json") as f:
        paper_meta = json.load(f)
    meta = paper_meta[str(paper_id)]

Confirmed join key: df.iloc[int(paper_id)] -- POSITIONAL, not .loc[].
The dataframe's own .index is not a unique paper_id (see df.info():
"Index: 171880 entries, 0 to 10"), so .loc[] would silently return the
wrong rows or raise on duplicate labels. This script always uses
.iloc[], matching what you verified manually.

Usage:
    python build_paper_metadata_lookup.py \
        --provenance PROVENANCE/hf_to_original_map.json \
        --pickle DATA/ED4RE_2603.pickle \
        --output PROVENANCE/paper_metadata.json
"""

import argparse
import json

import pandas as pd

# Content and References are large (full paper text / full bibliography)
# and rarely needed just for provenance/citation purposes -- opt in
# explicitly with --include-heavy if you want them in the output too.
DEFAULT_FIELDS = [
    "Title", "Authors", "Authors_and_Affiliations", "Affiliations",
    "DOI", "Journal", "Date", "Subjects", "Keywords",
]
HEAVY_FIELDS = ["Content", "References", "Abstract"]


def json_safe(value):
    """Coerce pandas/numpy scalars and containers into plain JSON-serializable types."""
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar (e.g. numpy.int64, numpy.str_)
        return value.item()
    return value


def collect_paper_ids(provenance_path):
    with open(provenance_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    paper_ids = set()
    for rec in records:
        for cand in rec.get("candidates", []):
            pid = cand.get("paper_id")
            if pid is not None:
                paper_ids.add(str(pid))
    return paper_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provenance", required=True,
                         help="Path to hf_to_original_map.json from build_hf_provenance_map.py")
    parser.add_argument("--pickle", required=True, help="Path to ED4RE_2603.pickle")
    parser.add_argument("--output", default="paper_metadata.json")
    parser.add_argument("--include-heavy", action="store_true",
                         help="Also include Content, References, Abstract (much larger output file).")
    args = parser.parse_args()

    paper_ids = collect_paper_ids(args.provenance)
    print(f"Found {len(paper_ids)} unique paper_id(s) referenced in {args.provenance}")

    df = pd.read_pickle(args.pickle)
    print(f"Loaded pickle with {len(df)} rows (using positional df.iloc[paper_id], not .loc[])")

    fields = DEFAULT_FIELDS + (HEAVY_FIELDS if args.include_heavy else [])

    metadata = {}
    missing = []
    for pid_str in sorted(paper_ids, key=lambda x: (len(x), x)):
        try:
            pid = int(pid_str)
        except ValueError:
            missing.append(pid_str)
            continue

        if pid < 0 or pid >= len(df):
            missing.append(pid_str)
            continue

        row = df.iloc[pid]
        metadata[pid_str] = {field: json_safe(row[field]) for field in fields if field in row}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata for {len(metadata)} papers to: {args.output}")
    if missing:
        preview = missing[:20]
        suffix = " ..." if len(missing) > 20 else ""
        print(f"WARNING: {len(missing)} paper_id(s) could not be resolved against the pickle "
              f"(out of range or non-numeric): {preview}{suffix}")


if __name__ == "__main__":
    main()