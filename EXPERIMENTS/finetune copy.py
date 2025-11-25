import argparse
import json
import re
import torch
from pathlib import Path
from datasets import load_dataset

# --- Helper Functions ---

def shorten_hf_name(hf_id):
    """Transforms a HF model ID into a short, filename-safe string."""
    name = hf_id.split("/")[-1]
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = re.sub(r"(_hf|_model)$", "", name, flags=re.IGNORECASE)
    return name.strip("_")

def load_training_config(json_path):
    """Loads JSON config with fallback defaults."""
    default_config = {
        "learning_rate": 5e-5,
        "batch_size": 16,
        "num_epochs": 5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "save_total_limit": 3,
        "eval_steps": 500,
        "save_steps": 500,
        "fp16": True,
        "gliner_specific": {},
        "spanmarker_specific": {}
    }
    
    if json_path:
        with open(json_path, 'r') as f:
            user_config = json.load(f)
            # Deep merge dictionaries
            for key, val in user_config.items():
                if isinstance(val, dict) and key in default_config:
                    default_config[key].update(val)
                else:
                    default_config[key] = val
                    
    return default_config

# --- Main Setup ---

parser = argparse.ArgumentParser(description='Fine-tune NER models (GLiNER or SpanMarker).')
parser.add_argument("--model_type", type=str, required=True, choices=["GLINER", "SPANMARKER"])
parser.add_argument("--dataset_id", type=str, required=True)
parser.add_argument("--model_id", type=str, required=True)
parser.add_argument("--config_file", type=str, default="config.json", help="Path to hyperparameters JSON")

args = parser.parse_args()

# 1. Device Setup
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("CUDA not available. Using CPU.")

# 2. Load Config & Data
hyperparams = load_training_config(args.config_file)
dataset = load_dataset(args.dataset_id)
labels = dataset["train"].features["ner_tags"].feature.names

print(f"Loaded dataset: {args.dataset_id} | Labels: {len(labels)}")

# 3. Define Unified Output Path
# Format: models/TYPE/ModelName_DatasetName
short_model = shorten_hf_name(args.model_id)
short_data = shorten_hf_name(args.dataset_id)
output_name = f"{short_model}_{short_data}"
output_dir = Path("models") / args.model_type / output_name

print(f"Output directory set to: {output_dir}")
print(f"Starting fine-tuning for {args.model_type}...")

# --- Model Specific Logic ---

if args.model_type == "GLINER":
    from gliner import GLiNER
    from gliner.training import Trainer as GlinerTrainer, TrainingArguments as GlinerArgs
    from gliner.data_processing.collator import DataCollator
    from dataset_processing import hf_dataset_to_gliner_format

    # Prepare Data
    train_dataset = hf_dataset_to_gliner_format(dataset["train"], labels)
    test_dataset = hf_dataset_to_gliner_format(dataset["validation"], labels)
    
    # Load Model
    model = GLiNER.from_pretrained(args.model_id)
    model.to(device)
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

    # GLiNER Specific Config extraction
    g_conf = hyperparams.get("gliner_specific", {})

    training_args = GlinerArgs(
        output_dir=str(output_dir),
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
        others_lr=g_conf.get("others_lr", 1e-5),
        others_weight_decay=hyperparams["weight_decay"],
        lr_scheduler_type="linear",
        warmup_ratio=hyperparams["warmup_ratio"],
        per_device_train_batch_size=hyperparams["batch_size"],
        per_device_eval_batch_size=hyperparams["batch_size"],
        focal_loss_alpha=g_conf.get("focal_loss_alpha", 0.75),
        focal_loss_gamma=g_conf.get("focal_loss_gamma", 2),
        num_train_epochs=hyperparams["num_epochs"],
        eval_strategy="steps",
        save_steps=hyperparams["save_steps"],
        save_total_limit=hyperparams["save_total_limit"],
        use_cpu=False,
        report_to="none",
    )

    trainer = GlinerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    print("GLiNER training complete.")

elif args.model_type == "SPANMARKER":
    from transformers import TrainingArguments, AutoConfig
    from span_marker import SpanMarkerModel, Trainer, SpanMarkerModelCardData

    sm_conf = hyperparams.get("spanmarker_specific", {})

    # Model Card Metadata
    model_card = SpanMarkerModelCardData(
        model_id=f"{short_model}-{short_data}",
        encoder_id=args.model_id,
        dataset_name=short_data,
        dataset_id=args.dataset_id,
        language="en",
    )

    # Load Model
    model = SpanMarkerModel.from_pretrained(
        args.model_id,
        labels=labels,
        model_max_length=sm_conf.get("model_max_length", 256),
        marker_max_length=sm_conf.get("marker_max_length", 128),
        entity_max_length=sm_conf.get("entity_max_length", 14),
        model_card_data=model_card,
    )

    # Fix for some BERT-based models in SpanMarker
    if not hasattr(model.config, "encoder") or model.config.encoder is None:
        model.config.encoder = AutoConfig.from_pretrained(args.model_id)

    # HF Training Arguments
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=hyperparams["learning_rate"],
        per_device_train_batch_size=hyperparams["batch_size"],
        per_device_eval_batch_size=hyperparams["batch_size"],
        num_train_epochs=hyperparams["num_epochs"],
        weight_decay=hyperparams["weight_decay"],
        warmup_ratio=hyperparams["warmup_ratio"],
        fp16=hyperparams.get("fp16", True),
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=hyperparams["eval_steps"],
        save_total_limit=hyperparams["save_total_limit"],
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()
    
    trainer.save_model(output_dir / "checkpoint-final")
    print("SpanMarker training complete.")