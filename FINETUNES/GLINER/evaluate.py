#!/usr/bin/env python
# coding: utf-8

# Script do develop evaluation procedure.

# In[3]:


import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments as GlinerArgs
from gliner.data_processing.collator import DataCollator

from datasets import load_dataset

from dataset_processing import CLIRENER_LABELS_V1, transform_to_ner_format


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


# In[6]:


model_path_or_id = "models/GLINER_med_v2_5/checkpoint-final"
model = GLiNER.from_pretrained(model_path_or_id)


# In[7]:


model_predictions = []
for row in dataset["test"]:
    text = row["text"]
    labels = list(CLIRENER_LABELS_V1)

    entities = model.predict_entities(text, labels, threshold=0.1)
    model_predictions.append({
        "text": text,
        "entities": entities
    })

model_predictions_transformed = transform_to_ner_format(model_predictions, labels)

model_predictions_ids = []
for row in model_predictions_transformed[0]:
    model_predictions_ids.append(row["ner_tags"])


# In[9]:


from nervaluate import Evaluator

labels = dataset["train"].features["ner_tags"].feature.names
true = ids_to_labels(TRUE, labels)
  
pred = ids_to_labels(model_predictions_ids, labels)
   
evaluator = Evaluator(true, pred, tags=list(CLIRENER_LABELS_V1), loader="list")


# In[10]:


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


# In[11]:


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

