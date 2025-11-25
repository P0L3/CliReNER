import argparse
import json
import torch
import re
import os
from pathlib import Path
from datasets import load_dataset
from transformers import TrainingArguments

from dataset_processing import hf_dataset_to_gliner_format


def shorten_name(name):
    """Transforms a Hugging Face ID into a clean, filename-safe string."""
    name = name.split("/")[-1]
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

def get_output_dir(base_dir, model_type, model_id, dataset_id):
    """Unified naming convention: models/TYPE/ModelName_DatasetName"""
    m_name = shorten_name(model_id)
    d_name = shorten_name(dataset_id)
    return Path(base_dir) / model_type / f"{m_name}_{d_name}"

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def train_gliner(model_id, dataset, labels, config, output_dir, device):
    from gliner import GLiNER
    from gliner.training import Trainer, TrainingArguments as GlinerArgs
    from gliner.data_processing.collator import DataCollator
    
    print("--- Initializing GLiNER Training ---")
    
    # Process Datasets
    train_ds = hf_dataset_to_gliner_format(dataset["train"], labels)
    val_ds = hf_dataset_to_gliner_format(dataset["validation"], labels)
    # test_ds = hf_dataset_to_gliner_format(dataset["test"], labels) 

    model = GLiNER.from_pretrained(model_id)
    model.to(device)
    
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)
    
    # Extract params from JSON
    train_params = config.get("training_parameters", {})
    
    # Dynamic epoch calculation logic 
    if train_params.get("calculate_epochs_from_steps", False):
        target_steps = train_params.get("target_steps", 4000)
        batch_size = train_params.get("per_device_train_batch_size", 8)
        num_batches = len(train_ds) // batch_size
        num_epochs = max(1, target_steps // num_batches)
        train_params["num_train_epochs"] = num_epochs
        # Remove custom keys so they don't crash TrainingArguments
        del train_params["calculate_epochs_from_steps"]
        del train_params["target_steps"]

    # Initialize Training Arguments
    # We update output_dir explicitly to ensure uniformity
    training_args = GlinerArgs(
        output_dir=str(output_dir),
        **train_params
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    # Save final model explicitly in the unified format
    model.save_pretrained(output_dir / "checkpoint-final")

def train_spanmarker(model_id, dataset, labels, config, output_dir, device):
    from span_marker import SpanMarkerModel, Trainer, SpanMarkerModelCardData
    from transformers import AutoConfig

    print("--- Initializing SpanMarker Training ---")

    model_params = config.get("model_parameters", {})
    train_params = config.get("training_parameters", {})
    
    # Model Card Data
    dataset_name = shorten_name(dataset.builder_name if dataset.builder_name else "dataset")
    card_data = SpanMarkerModelCardData(
        model_id=f"{shorten_name(model_id)}-{dataset_name}",
        encoder_id=model_id,
        dataset_id=dataset_name,
        license="cc-by-sa-4.0",
        language="en",
    )

    # Initialize Model
    model = SpanMarkerModel.from_pretrained(
        model_id,
        labels=labels,
        model_card_data=card_data,
        **model_params # e.g. model_max_length, marker_max_length, entity_max_length
    )

    # Ensure encoder config exists
    if not hasattr(model.config, "encoder") or model.config.encoder is None:
        model.config.encoder = AutoConfig.from_pretrained(model_id)

    # Initialize Training Arguments
    args = TrainingArguments(
        output_dir=str(output_dir),
        **train_params
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()
    
    trainer.save_model(output_dir / "checkpoint-final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune NER models (GLiNER or SpanMarker) using JSON configs.')
    
    parser.add_argument("--model_type", type=str, required=True, choices=["GLINER", "SPANMARKER"])
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True, help="Path to the JSON configuration file")
    
    args = parser.parse_args()

    # 1. Device Setup
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda:0')
    else:
        print("CUDA not available. Using CPU.")
        device = torch.device('cpu')

    # 2. Load Config
    config = load_config(args.config_path)

    # 3. Load Data
    print(f"Loading dataset: {args.dataset_id}")
    dataset = load_dataset(args.dataset_id)
    
    # Extract labels (Assumes standard NER format)
    if "ner_tags" in dataset["train"].features:
        labels = dataset["train"].features["ner_tags"].feature.names
    else:
        # Fallback or custom logic if your dataset structure differs
        labels = [] 
        print("Warning: Could not automatically detect 'ner_tags' feature names.")

    print(f"Labels found: {len(labels)}")

    # 4. Determine Output Path
    output_dir = get_output_dir("models", args.model_type, args.model_id, args.dataset_id)
    print(f"Output directory set to: {output_dir}")

    # 5. Run Training
    if args.model_type == "GLINER":
        train_gliner(args.model_id, dataset, labels, config, output_dir, device)
    elif args.model_type == "SPANMARKER":
        train_spanmarker(args.model_id, dataset, labels, config, output_dir, device)