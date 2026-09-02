"""
gliner_2stage_tune.py

Two-stage curriculum trainer for a custom-backbone GLiNER (v1, urchade/GLiNER)
model. Replaces EXPERIMENTS/gliner2_2stage_tune.py -- ported to the `gliner`
library because gliner2 checkpoints cannot be loaded back via `gliner`
(confirmed directly). A `gliner`-native custom encoder IS supported by the
library itself, per urchade:

    https://github.com/urchade/GLiNER/issues/262#issuecomment-4651561017

    from gliner import GLiNER
    model = GLiNER.from_config({
        "model_name": "<any HF encoder>", "name": "my model", "max_width": 12,
        "hidden_size": 768, "dropout": 0.3, "fine_tune": True,
        "subtoken_pooling": "first", "span_mode": "markerV0",
        "max_types": 100, "max_len": 512,
    })

    Stage 1 (General):    PileNER                       -> pilener_2025_gliner.json
    Stage 2 (Related CC): IBMCCNER+BioDivNER+ClimateIE   -> pile_ccner_2025_gliner.json

Both files are plain JSON arrays of {"tokenized_text": [...], "ner": [[start,
end_inclusive, label], ...]} dicts -- GLiNER v1's native training shape,
produced by gliner_custom_dataset.py (the GLiNER v1 equivalent of
FINETUNES/GLINER/gliner2_custom_dataset.ipynb).

Only stages 1 and 2 are handled (no CliReNER-specific stage 3, and no
entity_descriptions -- that's a GLiNER2-only feature with no GLiNER v1
equivalent). Checkpoints chain: stage 2 initializes from stage 1's
checkpoint-final. A slice of stage 1's train split can optionally be
replayed into stage 2's training data.

Note on hidden_size: GLiNERConfig does NOT auto-sync hidden_size to the
actual encoder -- it's used as-given to size the task-specific layers. This
script pulls it from AutoConfig.from_pretrained(BACKBONE_MODEL).hidden_size
automatically so it can never silently mismatch.

Follows EXPERIMENTS/finetune.py's train_gliner() pattern: gliner.training
Trainer/TrainingArguments/DataCollator, report_to="wandb", final save always
to `checkpoint-final`.

--------------------------------------------------------------------------
FIXED (this version) -- root-caused against the actual installed gliner
package source, not guessed:
  1. Removed the "rename tokenized_text -> tokens" block in run_stage().
     It was backwards: gliner.data_processing.processor.collate_raw_batch()
     reads b["tokenized_text"] directly (confirmed from source), and both
     SpanDataCollator/TokenDataCollator's docstrings require 'tokenized_text'
     and 'ner' on each raw example. "tokens" is only the INTERNAL batch key
     used after preprocess_example() runs -- not the raw input key. The
     rename was deleting the key the collator needs, raising
     KeyError: 'tokenized_text' during collation.
  2. Restored tokenizer=model.data_processor.transformer_tokenizer on the
     Trainer(...) call, matching EXPERIMENTS/finetune.py's proven pattern.
     Not fatal without it (gliner's Trainer._save() guards with
     getattr(self, "processing_class", None) or getattr(self, "tokenizer",
     None)), but intermediate epoch/step checkpoints wouldn't get the
     tokenizer saved alongside them without it.
  3. Simplified the DataCollator import: try the bare `DataCollator` name
     first (proven to exist in your environment, since finetune.py imports
     it successfully there), falling back to `SpanDataCollator` for newer
     gliner versions where it's been renamed. Dropped the
     `DataCollatorWithPadding` branch -- confirmed that name doesn't exist
     anywhere in gliner.data_processing.collator, so it was dead code.
--------------------------------------------------------------------------

Usage: edit the CONFIG block below, then:
    python gliner_2stage_tune.py
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

try:
    from EXPERIMENTS.finetune import shorten_name  # type: ignore
except Exception:
    import re
    def shorten_name(name):
        name = name.split("/")[-1]
        name = re.sub(r"[^a-zA-Z0-9]", "_", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_")


# =====================================================================
# Configuration Variables (edit these directly instead of CLI args)
# =====================================================================

BACKBONE_MODEL = "P0L3/clirebert_clirevocab_uncased"   # only used for Stage 1 cold start
RUN_NAME = "clirebert_clirevocab_uncased_6estage1"                                   # None -> shorten_name(BACKBONE_MODEL)

STAGE1_DATA = "./FINETUNES/GLINER/pilener_2025_gliner.json"
STAGE2_DATA = "./FINETUNES/GLINER/pile_ccner_2025_gliner.json"

RUN_STAGES = [2]          # subset e.g. [2] to resume from an existing stage-1 checkpoint
OUTPUT_BASE = Path("EXPERIMENTS/models/GLINER_CUSTOM")
SEED = 301202

WANDB_PROJECT = "clirener-gliner-backbone-swap"
WANDB_ENTITY = None

# If set, the FIRST stage in RUN_STAGES loads from this checkpoint instead of
# a fresh backbone (stage 1) or an auto-detected previous stage (stage 2).
INIT_FROM_CHECKPOINT = None

CHECK_DATA_ONLY = False       # True: just load/count the JSON files, no torch/gliner import
SMOKE_TEST = False            # True: 20 train / 10 eval samples, 1 epoch, no eval, no wandb

MAX_SUBWORD_TOKENS = 270 

# GLiNER model_cfg fixed fields (hidden_size is filled in at runtime via
# AutoConfig, since GLiNERConfig does not auto-sync it -- see docstring)
MODEL_CFG_BASE = {
    "max_width": 12,
    "dropout": 0.3,
    "fine_tune": True,
    "subtoken_pooling": "first",
    "span_mode": "markerV0",
    "max_types": 100,
    "max_len": 512,
}

# Per-stage training_parameters map 1:1 onto gliner.training.TrainingArguments
# kwargs (same keys as EXPERIMENTS/gliner_config.json). Values below are
# illustrative placeholders mirroring EXPERIMENTS/finetune.py's existing
# runs -- not independently tuned for this backbone/data mix, adjust freely.

STAGE_CONFIG = {
    1: {
        "eval_ratio": 0.02,
        "replay_fraction": 0.0,
        "training_parameters": {
            "learning_rate": 1e-5,               # Translated from encoder_lr
            "others_lr": 5e-4,                   # Translated from task_lr
            "weight_decay": 0.01,
            "others_weight_decay": 0.01,         # Added to match task_lr behavior
            "lr_scheduler_type": "cosine",       # Translated from scheduler_type
            "warmup_ratio": 0.1,
            "per_device_train_batch_size": 16,   # Translated from batch_size
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "fp16": True,                        # Enabled from JSON
            "num_train_epochs": 6, # 3,               # Translated from num_epochs
            "eval_strategy": "epoch",            # Matched from JSON
            "save_strategy": "epoch",            # Must match eval_strategy for load_best_model
            "save_total_limit": 2,
            "load_best_model_at_end": True,      # Replaces save_best
            "metric_for_best_model": "eval_loss",
            "logging_steps": 50,
            "focal_loss_alpha": 0.75,            # Kept from your original script
            "focal_loss_gamma": 2,               # Kept from your original script
            "dataloader_num_workers": 0,
            "use_cpu": False,
        },
        "early_stopping_patience": 3,            # Handled in the trainer setup below
    },
    2: {
        "eval_ratio": 0.05,
        "replay_fraction": 0.05,
        "training_parameters": {
            "learning_rate": 1e-5,
            "others_lr": 5e-4,
            "weight_decay": 0.01,
            "others_weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "fp16": True,
            "num_train_epochs": 6,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "logging_steps": 20,
            "focal_loss_alpha": 0.75,
            "focal_loss_gamma": 2,
            "dataloader_num_workers": 0,
            "use_cpu": False,
        },
        "early_stopping_patience": 3,
    },
}

STAGE_DATA_PATHS = {1: STAGE1_DATA, 2: STAGE2_DATA}
STAGE_LABELS = {1: "general (PileNER)", 2: "related CC NER (IBMCCNER+BioDivNER+ClimateIE)"}

if RUN_NAME is None:
    RUN_NAME = shorten_name(BACKBONE_MODEL)


# =====================================================================
# Data loading (torch/gliner-free -- safe to run for CHECK_DATA_ONLY)
# =====================================================================

def load_gliner_json(path):
    print(f"[data] Loading {path}")
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    print(f"[data] {len(examples)} examples")
    return examples


def summarize(examples, label=""):
    zero_entity = sum(1 for ex in examples if not ex.get("ner"))
    label_counts = defaultdict(int)
    for ex in examples:
        for span in ex.get("ner", []):
            label_counts[span[2]] += 1
    print(f"[data:{label}] total={len(examples)} zero_entity={zero_entity} "
          f"unique_labels={len(label_counts)}")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {lbl:<30}: {cnt}")


def split_train_eval(examples, eval_ratio, seed):
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_ratio)) if eval_ratio > 0 else 0
    eval_data = shuffled[:n_eval]
    train_data = shuffled[n_eval:]
    return train_data, eval_data


# =====================================================================
# Output-dir / checkpoint helpers
# =====================================================================

def get_stage_dir(output_base, run_name, stage_num, seed):
    return output_base / run_name / f"stage{stage_num}_s{seed}"


def checkpoint_path(stage_dir):
    """EXPERIMENTS/finetune.py's convention: the final save is always checkpoint-final."""
    return stage_dir / "checkpoint-final"


# =====================================================================
# Model construction
# =====================================================================

def build_fresh_model(backbone_model, model_cfg_base):
    from gliner import GLiNER
    from transformers import AutoConfig

    hidden_size = AutoConfig.from_pretrained(backbone_model).hidden_size
    print(f"  [model] AutoConfig.from_pretrained('{backbone_model}').hidden_size = {hidden_size}")

    model_cfg = dict(model_cfg_base)
    model_cfg["model_name"] = backbone_model
    model_cfg["name"] = RUN_NAME
    model_cfg["hidden_size"] = hidden_size

    print(f"  [model] Building fresh GLiNER model via GLiNER.from_config({model_cfg})")
    return GLiNER.from_config(model_cfg)


def load_model_from_checkpoint(ckpt_path):
    from gliner import GLiNER
    print(f"  [model] Loading GLiNER model from checkpoint: {ckpt_path}")
    return GLiNER.from_pretrained(str(ckpt_path))


# =====================================================================
# Stage execution
# =====================================================================

def run_stage(stage_num, model, stage_cfg, data_path, output_dir, seed,
              wandb_project, wandb_entity, run_name, replay_pool=None, smoke_test=False):
    import torch
    import wandb
    from gliner.training import Trainer, TrainingArguments as GlinerArgs

    # --- DataCollator import: try the name proven to work in your
    # environment (finetune.py imports this exact bare name successfully
    # there) first, falling back to the newer-gliner rename. There is no
    # DataCollatorWithPadding anywhere in gliner.data_processing.collator --
    # confirmed from source -- so that branch was dead code and is dropped.
    try:
        from gliner.data_processing.collator import DataCollator
    except ImportError:
        from gliner.data_processing.collator import SpanDataCollator as DataCollator

    print(f"\n{'#' * 70}\nSTAGE {stage_num}: {STAGE_LABELS[stage_num]}\n{'#' * 70}")

    tokenizer = getattr(model.data_processor, "transformer_tokenizer", None)
    examples = load_gliner_json(data_path)

    if tokenizer is not None:
        kept, dropped = [], 0
        for ex in tqdm(examples):
            n_subwords = sum(len(tokenizer.tokenize(tok)) for tok in ex["tokenized_text"])
            if n_subwords <= MAX_SUBWORD_TOKENS:
                kept.append(ex)
            else:
                # print(ex)
                dropped += 1
        if dropped:
            print(f"  [length filter] dropped {dropped}/{len(examples)} example(s) "
                  f"exceeding {MAX_SUBWORD_TOKENS} subword tokens")
        examples = kept
    else:
        print("  [length filter] WARNING: model.data_processor.transformer_tokenizer "
              "not found -- skipping length filter, sequences over 512 subwords "
              "(after GLiNER's schema prompt) will crash the encoder")

    summarize(examples, label=f"stage{stage_num}")

    eval_ratio = float(stage_cfg.get("eval_ratio", 0.05))
    train_data, eval_data = split_train_eval(examples, eval_ratio, seed)

    # NOTE: no key renaming here. Both train_data and eval_data must keep
    # "tokenized_text" (not "tokens") -- see the FIXED note at the top of
    # this file for why the previous rename broke the collator.

    # --- light experience replay: mix in a sample of the previous stage's
    # TRAIN examples (never its eval split) into this stage's training set.
    replay_fraction = float(stage_cfg.get("replay_fraction", 0.0))
    if replay_pool and replay_fraction > 0:
        n_replay = int(len(train_data) * replay_fraction)
        replay_sample = random.Random(seed).sample(replay_pool, min(n_replay, len(replay_pool)))
        train_data = train_data + replay_sample
        random.Random(seed).shuffle(train_data)
        print(f"  Replay: mixed in {len(replay_sample)} example(s) from the previous "
              f"stage's train split ({replay_fraction:.1%} target fraction)")

    if smoke_test:
        train_data = train_data[:20]
        eval_data = eval_data[:10]
        print("  [smoke_test] truncated to 20 train / 10 eval examples")

    print(f"  Train: {len(train_data)} | Eval: {len(eval_data)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)
    except TypeError:
        data_collator = DataCollator(model.config)

    train_params = dict(stage_cfg.get("training_parameters", {}))

    # Dynamic epoch calculation (same optional feature as EXPERIMENTS/finetune.py)
    if train_params.get("calculate_epochs_from_steps", False):
        target_steps = train_params.get("target_steps", 4000)
        batch_size = train_params.get("per_device_train_batch_size", 8)
        num_batches = max(1, len(train_data) // batch_size)
        train_params["num_train_epochs"] = max(1, target_steps // num_batches)
        del train_params["calculate_epochs_from_steps"]
        del train_params["target_steps"]

    if smoke_test:
        train_params["num_train_epochs"] = 1
        train_params["eval_strategy"] = "no"
        train_params["report_to"] = "none"
    else:
        train_params["report_to"] = "wandb" if wandb_project else "none"
    train_params["seed"] = seed

    training_args = GlinerArgs(output_dir=str(output_dir), **train_params)

    run = None
    if train_params["report_to"] == "wandb":
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"{run_name}_stage{stage_num}_s{seed}",
            reinit=True,
            config={"stage": stage_num, "seed": seed, **stage_cfg},
        )

    callbacks = []
    patience = stage_cfg.get("early_stopping_patience", 0)
    if patience > 0 and train_params.get("load_best_model_at_end"):
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    # Matches EXPERIMENTS/finetune.py's train_gliner(): pass the model's own
    # tokenizer explicitly so intermediate checkpoint saves during training
    # (not just the final manual save below) include it. Not fatal without
    # it -- gliner's Trainer._save() guards for a missing tokenizer -- but
    # cheap to keep consistent with the proven-working pattern.
    tokenizer = getattr(model.data_processor, "transformer_tokenizer", None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data if len(eval_data) else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()

    final_ckpt = checkpoint_path(output_dir)
    model.save_pretrained(final_ckpt)
    print(f"  Stage {stage_num} done. Checkpoint: {final_ckpt}")

    if run is not None:
        wandb.finish()

    return final_ckpt, train_data


# =====================================================================
# Main
# =====================================================================

def main():
    stages = [s for s in RUN_STAGES if s in (1, 2)]
    stages = [s for s in stages if STAGE_DATA_PATHS.get(s)]
    if not stages:
        print("No valid stages to run (check RUN_STAGES / STAGE_DATA_PATHS).")
        return

    if CHECK_DATA_ONLY:
        print("=== CHECK_DATA_ONLY: loading/summarizing JSON files, no training ===")
        for s in stages:
            examples = load_gliner_json(STAGE_DATA_PATHS[s])
            summarize(examples, label=f"stage{s}")
        return

    model = None
    replay_pool = None
    prev_checkpoint = Path(INIT_FROM_CHECKPOINT) if INIT_FROM_CHECKPOINT else None

    for stage_num in stages:
        stage_cfg = STAGE_CONFIG[stage_num]
        stage_dir = get_stage_dir(OUTPUT_BASE, RUN_NAME, stage_num, SEED)

        if model is None:
            if prev_checkpoint is not None:
                model = load_model_from_checkpoint(prev_checkpoint)
            elif stage_num == 1:
                model = build_fresh_model(BACKBONE_MODEL, MODEL_CFG_BASE)
            else:
                prev_dir = get_stage_dir(OUTPUT_BASE, RUN_NAME, stage_num - 1, SEED)
                found = checkpoint_path(prev_dir)
                if not found.exists():
                    raise RuntimeError(
                        f"Stage {stage_num} requested without INIT_FROM_CHECKPOINT, and no "
                        f"checkpoint found at {found}. Run stage {stage_num - 1} first, or "
                        f"set INIT_FROM_CHECKPOINT."
                    )
                model = load_model_from_checkpoint(found)

        prev_checkpoint, replay_pool = run_stage(
            stage_num=stage_num,
            model=model,
            stage_cfg=stage_cfg,
            data_path=STAGE_DATA_PATHS[stage_num],
            output_dir=stage_dir,
            seed=SEED,
            wandb_project=WANDB_PROJECT,
            wandb_entity=WANDB_ENTITY,
            run_name=RUN_NAME,
            replay_pool=replay_pool,
            smoke_test=SMOKE_TEST,
        )
        # Next stage re-loads from the saved checkpoint (fresh optimizer/scheduler)
        # rather than continuing to train the in-memory `model` object -- matches
        # EXPERIMENTS/finetune.py's per-run from_pretrained() pattern.
        model = None

    print(f"\n{'=' * 70}\nPIPELINE COMPLETE. Final checkpoint: {prev_checkpoint}\n{'=' * 70}")


if __name__ == "__main__":
    main()