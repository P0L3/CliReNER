import argparse
from datasets import load_dataset
import torch

from pathlib import Path

import re

def shorten_hf_name(hf_id):
    """
    Transforms a Hugging Face model ID into a short, filename-safe string.
    
    Example: 
        "P0L3/CliReNER_v_1_1_28_SILVER" -> "CliReNER_v_1_1_28_SILVER"
        "google-bert/bert-base-uncased" -> "bert_base_uncased"
    """
    # 1. Get the model name (remove the organization prefix)
    name = hf_id.split("/")[-1]
    
    # 2. Replace non-alphanumeric characters (-, ., spaces) with underscores
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    
    # 3. Collapse multiple underscores into one
    name = re.sub(r"_+", "_", name)
    
    # 4. Remove common redundant suffixes (optional, but cleaner)
    name = re.sub(r"(_hf|_model)$", "", name, flags=re.IGNORECASE)
    
    return name.strip("_")

parser = argparse.ArgumentParser(
                    prog='ModelFineTuning',
                    description='Program uses HuggingFace Dataset to fine-tune NER models.',
                    epilog='...')

parser.add_argument("--model_type", type=str)
parser.add_argument("--dataset_id", type=str)
parser.add_argument("--model_id", type=str)

args = parser.parse_args()

# Check CUDA
if torch.cuda.is_available():
    print(f"CUDA is available. Using {torch.cuda.device_count()} GPU(s).")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda:0')
else:
    print("CUDA is not available. Training will run on CPU.")
    device = torch.device('cpu')

# Load data
dataset = load_dataset(args.dataset_id)
print(f"Loaded HF dataset: {args.dataset_id}.")

# Dataset Info
labels = dataset["train"].features["ner_tags"].feature.names
print(f"Dataset contains {len(labels)} labels: [\"{labels[0]}\", ..., \"{labels[len(labels)//2]}\", ..., \"{labels[-1]}\"]")
print(f"Found {len(dataset)} splits: {list(dataset.keys())}")
for split in dataset:
    print(f"\t{split}: {len(dataset[split])} rows")

# Load model backend (GLiNER or specific BER-based encoder)
dataset_name = shorten_hf_name(args.dataset_id)
model_name = shorten_hf_name(args.model_id)

print(f"Performing fine-tuning based on {args.model_type} ...")

if args.model_type == "GLINER":
    # import json
    # import random

    # from seqeval.metrics.sequence_labeling import get_entities
    # import re
    # from collections import Counter
    # # import matplotlib.pyplot as plt
    # import pandas as pd
    # from typing import List, Dict

    from dataset_processing import *
    # import os
    # os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

    from gliner import GLiNERConfig, GLiNER
    from gliner.training import Trainer, TrainingArguments
    from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
    # from gliner.utils import load_config_as_namespace
    # from gliner.data_processing import WordsSplitter, GLiNERDataset
    
    train_dataset = hf_dataset_to_gliner_format(dataset["train"], labels)
    test_dataset = hf_dataset_to_gliner_format(dataset["validation"], labels)
    
    model = GLiNER.from_pretrained(args.model_id)
    
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)
    
    model.to(device)
    
    
    # calculate number of epochs
    num_steps = 4000
    batch_size = 8
    data_size = len(train_dataset)
    num_batches = data_size // batch_size
    num_epochs = max(1, num_steps // num_batches)

    training_args = TrainingArguments(
        output_dir="models/GLiNER_med_v2_5/CliReNER_v_1_1_28_SILVER",
        learning_rate=5e-6,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear", #cosine
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2,
        num_train_epochs=num_epochs,
        eval_strategy="steps", # PC1, # evaluation_strategy="steps",
        save_steps = 100,
        save_total_limit=10,
        dataloader_num_workers = 0,
        use_cpu = False,
        report_to="none",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    
    
    print("GLINER")
    
elif args.model_type == "SPANMARKER":
    from transformers import TrainingArguments
    from span_marker import SpanMarkerModel, Trainer, SpanMarkerModelCardData

    from transformers import AutoConfig
    
    
    new_model_id = f"P0L3/span-marker-{model_name}-{dataset_name}_25612814_100"
    model = SpanMarkerModel.from_pretrained(
        args.model_id,
        labels=labels,
        # SpanMarker hyperparameters:
        model_max_length=256,
        marker_max_length=128,
        entity_max_length=14,
        # Model card arguments
        model_card_data=SpanMarkerModelCardData(
            model_id=new_model_id,
            encoder_id=args.model_id,
            dataset_name=dataset_name,
            dataset_id=args.dataset_id,
            license="cc-by-sa-4.0",
            language="en",
        ),
    )
    
    if not hasattr(model.config, "encoder") or model.config.encoder is None:
        model.config.encoder = AutoConfig.from_pretrained(args.model_id)
    
    # Prepare the 🤗 transformers training arguments
    output_dir = Path("models") / args.model_id
    args = TrainingArguments(
        output_dir=output_dir,
        # Training Hyperparameters:
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=100,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,  # Replace `bf16` with `fp16` if your hardware can't use bf16.
        # Other Training parameters
        logging_first_step=True,
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=3000,
        save_total_limit=5,
        dataloader_num_workers=2,
    )
    
    # Initialize the trainer using our model, training args & dataset, and train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    trainer.train()
    
    # Compute & save the metrics on the test set
    metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    trainer.save_metrics("test", metrics)
    
    # Save the final checkpoint
    trainer.save_model(output_dir / "checkpoint-final")
    
    print("SPANMARKER")