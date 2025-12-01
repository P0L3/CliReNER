#!/usr/bin/env python
# coding: utf-8

# Script to develop evaluation procedure

# In[3]:


from span_marker import SpanMarkerModel, Trainer, SpanMarkerModelCardData
from transformers import AutoConfig

import argparse
import json
import torch
import re
import os
from pathlib import Path
from datasets import load_dataset
from transformers import TrainingArguments

from dataset_processing import hf_dataset_to_gliner_format, CLIRENER_LABELS_V1, transform_to_ner_format


# In[4]:


def ids_to_labels(pred_id_seqs, label_list):
    """
    Convert sequences of prediction IDs into label sequences.

    Args:
        pred_id_seqs (list[list[int]]): e.g. model predictions [[2, 5, 0], ...]
        label_list (list[str]): label names from dataset, e.g. dataset["train"].features["ner_tags"].feature.names

    Returns:
        list[list[str]]: converted label sequences
    """
    return [[label_list[i] for i in seq] for seq in pred_id_seqs]


# In[5]:


dataset = load_dataset("P0L3/CliReNER_v_1_1_28_SILVER")
labels = dataset["train"].features["ner_tags"].feature.names

TRUE = dataset["test"]["ner_tags"]


# In[10]:


model_path_or_id = "models/P0L3/span-marker-P0L3/clirebert_clirevocab_uncased-CliReNER_v_0_0_26/checkpoint-final"
model = SpanMarkerModel.from_pretrained(model_path_or_id)


# In[11]:


import transformers
print(transformers.__version__)


# In[35]:


text =  [row["text"] for row in dataset["test"]]

entities_list = model.predict(text)


# In[46]:


labels


# In[ ]:


model_predictions = []
for i, row in enumerate(entities_list):
    row_text = text[i]
    # print(row)
    entities = []
    for entity in row:
        temp_dict = {
            'start': entity["char_start_index"],
            'end': entity["char_end_index"],
            'text': entity["span"],
            'label': entity["label"],
            'score': entity["score"]
        }
        entities.append(temp_dict)
        
    prediction = {
        "text": row_text,
        "entities": entities
    }
    
    model_predictions.append(prediction)
labels = list(CLIRENER_LABELS_V1)
model_predictions_transformed = transform_to_ner_format(model_predictions, labels)

model_predictions_ids = []
for row in model_predictions_transformed[0]:
    model_predictions_ids.append(row["ner_tags"])


# In[48]:


from nervaluate import Evaluator

labels = dataset["train"].features["ner_tags"].feature.names
true = ids_to_labels(TRUE, labels)
  
pred = ids_to_labels(model_predictions_ids, labels)
   
evaluator = Evaluator(true, pred, tags=list(CLIRENER_LABELS_V1), loader="list")


# In[49]:


results, results_by_tag, result_indices, result_indices_by_tag = evaluator.evaluate()

import pandas as pd

df = pd.DataFrame(results)
print(df)

print("\n##results##:\n")
print("Strict: ", results["strict"])
print("Exact:  ", results["exact"])
print("\##results_by_tag##:\n")
print(results_by_tag)
print("\##result_indices##:\n")
print(result_indices)
print("\##result_indices_by_tag##:\n")
print(result_indices_by_tag)


# In[50]:


import matplotlib.pyplot as plt

def plot_overall_metrics(results):
    metrics = ['strict', 'exact', 'ent_type', 'partial']
    precision = [results[m]['precision'] for m in metrics]
    recall = [results[m]['recall'] for m in metrics]
    f1 = [results[m]['f1'] for m in metrics]

    x = range(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width for i in x], precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar([i + width for i in x], f1, width, label='F1 Score')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Overall NER Evaluation Scores")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_overall_metrics(results)

