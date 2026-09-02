"""
Three-stage curriculum trainer for a custom-backbone GLiNER2 model.

    Stage 1 (General):      PileNER               -> pilener_2025.jsonl
    Stage 2 (Related CC):   IBMCCNER+BioDivNER+ClimateIE (interleaved, native
                             labels)               -> pile_ccner_2025.jsonl
    Stage 3 (Specific):     CliReNER SILVER (28-label, entity_descriptions
                             from definitions.txt)  -> not yet built

Checkpoints are chained: Stage 2 initializes from Stage 1's checkpoint,
Stage 3 from Stage 2's. A small fraction of the previous stage's TRAIN
examples is mixed into each later stage's training set (light experience
replay), per stage in the JSON config.

Verified against the installed `gliner2==2.0.0` source (training/data.py,
training/trainer.py, models/span/model.py) on 2026-09-01 -- not just the
tutorial docs. Two things worth knowing if your local gliner2 differs:

  1. On-disk JSONL schema is `{"input": text, "output": {"entities": {...},
     "entity_descriptions": {...}}}` -- NOT a flat `{"text":..., "entities":...}`
     shape. `load_and_normalize_jsonl()` below auto-detects and repairs files
     using the flat shape (writes a `.bak` backup first).
  2. `GLiNER2(ExtractorConfig(model_name=<any HF encoder>))` builds a model
     with that encoder directly -- no custom subclass needed for gliner2>=2.0.
     If your training env is pinned to an older gliner2 where this doesn't
     work, swap `build_fresh_model()` below for the `CustomExtractor` pattern
     from FINETUNES/GLINER/SCRIPT_VERSIONS/gliner2_custom_encoder.py instead.

Usage:
    # Fast smoke test (no GPU-heavy run, just checks the plumbing end-to-end)
    python -m EXPERIMENTS.finetune_gliner2_stages \
        --config_path EXPERIMENTS/gliner2_stages_config.json \
        --stages 1 --smoke_test

    # Just repair/validate the JSONL files, no training, no torch import
    python -m EXPERIMENTS.finetune_gliner2_stages \
        --config_path EXPERIMENTS/gliner2_stages_config.json \
        --check_data_only

    # Full pipeline
    python -m EXPERIMENTS.finetune_gliner2_stages \
        --config_path EXPERIMENTS/gliner2_stages_config.json \
        --stages 1,2,3 \
        --wandb_project clirener-gliner2-stages

    # Resume from stage 2 only (auto-finds stage 1's checkpoint)
    python -m EXPERIMENTS.finetune_gliner2_stages \
        --config_path EXPERIMENTS/gliner2_stages_config.json \
        --stages 2,3
"""

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- naming convention -------------------------------------------------
# Reuse the repo's existing shorten_name()/output-dir convention if this is
# run as part of the EXPERIMENTS package; fall back to a local copy so the
# script also works standalone.
try:
    from EXPERIMENTS.finetune import shorten_name  # type: ignore
except Exception:
    def shorten_name(name: str) -> str:
        name = name.split("/")[-1]
        name = re.sub(r"[^a-zA-Z0-9]", "_", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_")


DEFAULT_STAGE_DATA = {
    1: "FINETUNES/GLINER/pilener_2025.jsonl",
    2: "FINETUNES/GLINER/pile_ccner_2025.jsonl",
    3: None,  # CliReNER SILVER -> GLiNER2 format converter not built yet
}
STAGE_LABELS = {1: "general (PileNER)", 2: "related CC NER", 3: "CliReNER-specific"}


# =========================================================================
# Data loading / repair (torch-free -- safe to run without a training env)
# =========================================================================

def load_and_normalize_jsonl(path: Path, drop_invalid: bool = True):
    """
    Load a JSONL file into a gliner2 TrainingDataset, auto-detecting and
    repairing files that used a flat {"text":..., "entities":...} shape
    instead of the real on-disk schema {"input":..., "output": {"entities":...}}.

    Backs up the original file to `<path>.bak` before rewriting, so this is
    safe to run repeatedly / idempotently.
    """
    from gliner2.training.data import InputExample, TrainingDataset

    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f if ln.strip()]

    examples = []
    flat_detected = False
    bad_lines = 0
    for line_num, line in enumerate(raw_lines, 1):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            continue

        if "input" in data and "output" in data:
            examples.append(InputExample.from_dict(data))
        elif "text" in data:
            flat_detected = True
            examples.append(InputExample(
                text=data["text"],
                entities=data.get("entities") or {},
                entity_descriptions=data.get("entity_descriptions"),
            ))
        else:
            bad_lines += 1
            print(f"  [{path.name}] line {line_num}: unrecognized schema, "
                  f"keys={list(data.keys())} -- skipped")

    if bad_lines:
        print(f"  [{path.name}] skipped {bad_lines} unparseable/unrecognized line(s)")

    dataset = TrainingDataset(examples)
    report = dataset.validate(raise_on_error=False)
    if report["invalid"]:
        print(f"  [{path.name}] {report['invalid']}/{report['total']} examples fail "
              f"strict validation (entity mention not verbatim in text). "
              f"Sample: {report['errors'][:3]}")
        print("INVALID:")
        for i in report["invalid_indices"]:
            print(dataset[i])

        if drop_invalid:
            invalid = set(report["invalid_indices"])
            dataset = TrainingDataset([ex for i, ex in enumerate(dataset.examples) if i not in invalid])
            print(f"  [{path.name}] dropped {len(invalid)} invalid example(s), {len(dataset)} remain")

    if flat_detected:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"  [{path.name}] flat {{'text','entities'}} schema detected -- "
                  f"backed up original to {backup.name} and rewrote to canonical "
                  f"gliner2 schema ({{'input','output'}})")
        dataset.save(path, validate_first=False)

    return dataset


# =========================================================================
# Output-dir / checkpoint helpers
# =========================================================================

def get_stage_dir(output_base: Path, run_name: str, stage_num: int, seed: int) -> Path:
    return output_base / run_name / f"stage{stage_num}_s{seed}"


def find_checkpoint(stage_dir: Path) -> Optional[Path]:
    """Prefer best/, then final/, then the most recently modified checkpoint-* dir."""
    if not stage_dir.exists():
        return None
    for name in ("best", "final"):
        candidate = stage_dir / name
        if candidate.exists():
            return candidate
    candidates = sorted(
        (p for p in stage_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


# =========================================================================
# Stage execution
# =========================================================================

def run_stage(
    stage_num: int,
    model,
    stage_cfg: Dict[str, Any],
    data_path: Path,
    output_dir: Path,
    run_name: str,
    seed: int,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    replay_pool: Optional[List] = None,
    smoke_test: bool = False,
) -> Tuple[Path, List]:
    from gliner2.training.data import TrainingDataset
    from gliner2.training.trainer import TrainingConfig, GLiNER2Trainer
    from transformers import AutoTokenizer

    print(f"\n{'#' * 70}\nSTAGE {stage_num}: {STAGE_LABELS[stage_num]}\n{'#' * 70}")

    dataset = load_and_normalize_jsonl(data_path)
    # dataset.print_stats()
    print("  Filtering out sequences > 512 tokens to prevent BERT crashes...")
    # Attempt to grab the tokenizer from the model, otherwise load it manually
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model.config.model_name)
        except Exception:
            # Fallback to the default backbone
            tokenizer = AutoTokenizer.from_pretrained("P0L3/clirebert_clirevocab_uncased")

    kept_examples = []
    dropped = 0
    for ex in dataset.examples:
        # Tokenize text to check actual sequence length
        tokens = tokenizer.tokenize(ex.text)
        # 512 - 2 (leaving room for [CLS] and [SEP] special tokens) = 510 and 20 leaway
        if len(tokens) <= 300:
            kept_examples.append(ex)
        else:
            dropped += 1
            
    if dropped > 0:
        print(f"  [Warning] Dropped {dropped} long examples that exceeded the 512-token limit.")
        # Re-initialize the dataset with only the safe examples
        dataset = TrainingDataset(kept_examples)

    eval_ratio = float(stage_cfg.get("eval_ratio", 0.05))
    train_data, eval_data, _ = dataset.split(
        train_ratio=1.0 - eval_ratio, val_ratio=eval_ratio, test_ratio=0.0,
        shuffle=True, seed=seed,
    )

    # --- light experience replay: mix in a sample of the previous stage's
    # TRAIN examples (never its eval split) into this stage's training set.
    replay_fraction = float(stage_cfg.get("replay_fraction", 0.0))
    if replay_pool and replay_fraction > 0:
        n_replay = int(len(train_data) * replay_fraction)
        replay_sample = random.Random(seed).sample(replay_pool, min(n_replay, len(replay_pool)))
        train_data.add_many(replay_sample)
        random.Random(seed).shuffle(train_data.examples)
        print(f"  Replay: mixed in {len(replay_sample)} example(s) from the previous "
              f"stage's train split ({replay_fraction:.1%} target fraction)")

    print(f"  Train: {len(train_data)} | Eval: {len(eval_data)}")

    training_params = dict(stage_cfg.get("training_parameters", {}))

    if sys.platform == "win32":
        training_params["num_workers"] = 0
    if smoke_test:
        training_params.update(num_epochs=1, max_train_samples=20, max_eval_samples=10,
                                eval_strategy="no", save_best=False, logging_steps=1)
        print("  [smoke_test] overriding training_parameters for a fast plumbing check")

    training_config = TrainingConfig(
        output_dir=str(output_dir),
        experiment_name=f"{run_name}_stage{stage_num}",
        seed=seed,
        report_to_wandb=bool(wandb_project) and not smoke_test,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=f"{run_name}_stage{stage_num}_s{seed}",
        wandb_tags=[f"stage{stage_num}", run_name],
        **training_params,
    )

    trainer = GLiNER2Trainer(model=model, config=training_config)
    results = trainer.train(train_data=train_data, eval_data=eval_data if len(eval_data) else None)

    print(f"  Stage {stage_num} done. best_metric={results.get('best_metric')} "
          f"total_steps={results.get('total_steps')}")

    checkpoint = find_checkpoint(output_dir)
    if checkpoint is None:
        raise RuntimeError(f"Stage {stage_num}: no checkpoint found under {output_dir} after training")
    print(f"  Checkpoint: {checkpoint}")

    return checkpoint, train_data.examples


# =========================================================================
# Model construction
# =========================================================================

def build_fresh_model(backbone_model: str):
    """
    gliner2>=2.0: the base Extractor/GLiNER2 class already loads any HF
    encoder generically from config.model_name -- no subclass needed.
    See module docstring if your installed gliner2 predates this.
    """
    from gliner2 import GLiNER2
    from gliner2.model import ExtractorConfig

    print(f"  Building fresh model on backbone: {backbone_model}")
    config = ExtractorConfig(model_name=backbone_model, max_len=512)
    return GLiNER2(config)


def load_model_from_checkpoint(checkpoint_path: Path):
    from gliner2 import GLiNER2
    print(f"  Loading model from checkpoint: {checkpoint_path}")
    return GLiNER2.from_pretrained(str(checkpoint_path))


# =========================================================================
# Main
# =========================================================================

def parse_stages(spec: str) -> List[int]:
    stages = sorted(set(int(s) for s in spec.split(",") if s.strip()))
    for s in stages:
        if s not in (1, 2, 3):
            raise ValueError(f"Invalid stage {s}; must be 1, 2, or 3")
    return stages


def main():
    parser = argparse.ArgumentParser(description="Three-stage GLiNER2 custom-backbone trainer")
    parser.add_argument("--config_path", type=str, required=True,
                         help="JSON file with per-stage training_parameters/eval_ratio/replay_fraction")
    parser.add_argument("--backbone_model", type=str, default="P0L3/clirebert_clirevocab_uncased")
    parser.add_argument("--stage1_data", type=str, default=DEFAULT_STAGE_DATA[1])
    parser.add_argument("--stage2_data", type=str, default=DEFAULT_STAGE_DATA[2])
    parser.add_argument("--stage3_data", type=str, default=DEFAULT_STAGE_DATA[3])
    parser.add_argument("--stages", type=str, default="1,2,3", help="Comma-separated subset, e.g. '2,3'")
    parser.add_argument("--output_base", type=str, default="EXPERIMENTS/models/GLINER2_CUSTOM")
    parser.add_argument("--run_name", type=str, default=None, help="Defaults to shorten_name(backbone_model)")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_from_checkpoint", type=str, default=None,
                         help="Override: start the first requested stage from this checkpoint "
                              "instead of a fresh backbone or an auto-detected previous stage")
    parser.add_argument("--check_data_only", action="store_true",
                         help="Only repair/validate the requested stages' JSONL files and print "
                              "stats -- no torch/model import, no training")
    parser.add_argument("--smoke_test", action="store_true",
                         help="Run each requested stage with 20 train / 10 eval samples, 1 epoch, "
                              "no eval/checkpointing -- verifies the pipeline end-to-end quickly")
    args = parser.parse_args()

    stages = parse_stages(args.stages)
    run_name = args.run_name or shorten_name(args.backbone_model)
    stage_data_paths = {1: args.stage1_data, 2: args.stage2_data, 3: args.stage3_data}

    with open(args.config_path, "r") as f:
        full_config = json.load(f)

    for s in stages:
        if stage_data_paths[s] is None:
            print(f"Stage {s} requested but no data path is set (--stage{s}_data). "
                  f"Skipping. Stage {s} = {STAGE_LABELS[s]}.")
    stages = [s for s in stages if stage_data_paths[s] is not None]
    if not stages:
        print("No stages left to run.")
        sys.exit(0)

    if args.check_data_only:
        print("=== --check_data_only: repairing/validating JSONL files, no training ===")
        for s in stages:
            load_and_normalize_jsonl(Path(stage_data_paths[s])).print_stats()
        return

    output_base = Path(args.output_base)
    model = None
    replay_pool: Optional[List] = None
    prev_checkpoint: Optional[Path] = Path(args.init_from_checkpoint) if args.init_from_checkpoint else None

    for stage_num in stages:
        stage_key = f"stage{stage_num}"
        if stage_key not in full_config:
            raise ValueError(f"--config_path is missing a '{stage_key}' entry")
        stage_cfg = full_config[stage_key]
        stage_dir = get_stage_dir(output_base, run_name, stage_num, args.seed)

        if model is None:
            if prev_checkpoint is not None:
                model = load_model_from_checkpoint(prev_checkpoint)
            elif stage_num == 1:
                model = build_fresh_model(args.backbone_model)
            else:
                # Resuming mid-pipeline: auto-find the previous stage's checkpoint.
                prev_dir = get_stage_dir(output_base, run_name, stage_num - 1, args.seed)
                found = find_checkpoint(prev_dir)
                if found is None:
                    raise RuntimeError(
                        f"Stage {stage_num} requested without --init_from_checkpoint, and no "
                        f"checkpoint was found for stage {stage_num - 1} under {prev_dir}. "
                        f"Run stage {stage_num - 1} first, or pass --init_from_checkpoint."
                    )
                model = load_model_from_checkpoint(found)
        # else: model already carried over from the previous loop iteration (in-memory chaining)

        prev_checkpoint, replay_pool = run_stage(
            stage_num=stage_num,
            model=model,
            stage_cfg=stage_cfg,
            data_path=Path(stage_data_paths[stage_num]),
            output_dir=stage_dir,
            run_name=run_name,
            seed=args.seed,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            replay_pool=replay_pool,
            smoke_test=args.smoke_test,
        )
        # Next stage re-loads from the saved checkpoint (fresh optimizer/scheduler,
        # matching gliner2's own "checkpoints are weights-only" design) rather than
        # continuing to train the in-memory `model` object.
        model = None

    print(f"\n{'=' * 70}\nPIPELINE COMPLETE. Final checkpoint: {prev_checkpoint}\n{'=' * 70}")


if __name__ == "__main__":
    main()