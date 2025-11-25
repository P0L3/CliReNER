import json
from datasets import load_dataset

from seqeval.metrics.sequence_labeling import get_entities

import re

from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Set
import os

import numpy as np
import datasets
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split



# ---------------- DATA CONFIGS
IBMCCNER_DIR = "ibm-research/Climate-Change-NER"
IBMCCNER_LABELS = {
    "climate-assets", "climate-datasets", "climate-greenhouse-gases",     
    "climate-hazards", "climate-impacts", "climate-mitigations",     
    "climate-models", "climate-nature", "climate-observations",     
    "climate-organisms", "climate-organizations", "climate-problem-origins",
    "climate-properties"
}

BIODIVNER_DIR = "/home/p0l3/RAD/CWED4ETA/CWED4ETA/CWED4ETA/DATA/BiodivNER"# "C:\\Users\\ANDRIJA_RAD\\CWED4ETA\\CWED4ETA\\DATA\\BiodivNER" #PC2 # "/home/p0l3/RAD/CWED4ETA/CWED4ETA/CWED4ETA/DATA/BiodivNER"  "C:\RAD\CWED4ETA\CWED4ETA\DATA\BiodivNER" PC1 
BIODIVNER_LABELS = {
    'Quality', 'Phenomena', 'Organism',
    'Matter', 'Location', 'Environment'
}

CLIRENER_DIR = "/home/p0l3/RAD/CLIRENER/CliReNER/DATA/LABEL_STUDIO/project-30-at-2025-11-14-12-19-2a7464a5.json"
# "C:\\Users\\ANDRIJA_RAD\\CLIRENER\\CliReNER\\DATA\\LABEL_STUDIO\\project-28-at-2025-11-11-12-33-f873727e.json"
# "C:\\Users\\ANDRIJA_RAD\\CWED4ETA\\CWED4ETA\\DATA\\CWED4ETA\\project-6-at-2025-10-15-07-02-c3e9e3bf.json" # PC2 
# "/home/p0l3/RAD/CWED4ETA/CWED4ETA/CWED4ETA/DATA/CWED4ETA/project-6-at-2025-10-15-07-27-c3e9e3bf.json"
CLIRENER_LABELS_V0 = {
    "Ecosystem", "Energy Source", "Natural Disaster", 
    "Meteorological Phenomenon", "Quantity", "Astronomical Object", 
    "Body of Water", "Disease", "Location", 
    "Physical Phenomenon", "Chemical", "Time Period", 
    "Organization", "Natural Phenomenon", "Field of Study", 
    "Mathematical Expression", "Measuring Device", "Geographical Feature", 
    "System", "Satellite", "Organism", 
    "Method", "Other", "Person", 
    "Artefact", "Body Part", "Symptom"
}

CLIRENER_LABELS_V1 = {
    "Ecosystem", "Energy Source", "Natural Disaster", 
    "Meteorological Phenomenon", "Quantity", "Intellectual Artefact", 
    "Body of Water", "Disease", "Location", 
    "Physical Phenomenon", "Chemical", "Time Period", 
    "Organization", "Natural Phenomenon", "Field of Study", 
    "Mathematical Expression", "Measuring Device", "Geographical Feature", 
    "System", "Satellite", "Organism", 
    "Method", "Other", "Person", 
    "Physical Artefact", "Body Part", "Asset",
    "Policy", 
}


CLIRENER_ANNOTATOR_IMPORTANCE = [2, 1, 3]


# ---------------- PREPREPROCESSING
def ibmccner_process_bio_documents(document_list, labels_to_keep):
    """
    Converts a list of documents in BIO line format into a structured
    dictionary with reconstructed text and character-level entity spans.
    """
    processed_data = []
    for doc_lines in document_list:
        doc_tokens = []
        doc_tags = []
        for line in doc_lines:
            if line.strip().startswith('-DOCSTART-') or not line.strip():
                continue
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                doc_tokens.append(parts[0])
                doc_tags.append(parts[1])
        if not doc_tokens:
            continue
        reconstructed_text = ""
        token_start_offsets = []
        for token in doc_tokens:
            token_start_offsets.append(len(reconstructed_text))
            reconstructed_text += token + " "
        reconstructed_text = reconstructed_text.rstrip()
        extracted_entities = get_entities(doc_tags)
        doc_entities = []
        for entity_label, start_token_idx, end_token_idx in extracted_entities:
            if entity_label in labels_to_keep:
                entity_text = " ".join(doc_tokens[start_token_idx : end_token_idx + 1])
                char_start = token_start_offsets[start_token_idx]
                last_token_idx = end_token_idx
                char_end = token_start_offsets[last_token_idx] + len(doc_tokens[last_token_idx])
                doc_entities.append({
                    "text": entity_text,
                    "label": entity_label,
                    "start": char_start,
                    "end": char_end
                })
        processed_data.append({
            "text": reconstructed_text,
            "entities": doc_entities
        })
    return processed_data

def biodivner_process_bio_documents(data_dir: str, labels_to_keep: Set[str], split = ["train", "validation"]) -> List[Dict]:
    """
    Reads BiodivNER CSV files, processes each sentence, and converts them into
    a structured format with reconstructed text and character-level entity spans.

    Args:
        data_dir: The directory containing the train.csv, dev.csv, and test.csv files.
        labels_to_keep: A set of entity labels to include in the output.

    Returns:
        A list of dictionaries, where each dictionary represents a sentence
        and has 'text' and 'entities' keys.
    """
    all_processed_sentences = []
    
    try:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"Error: Directory not found at '{data_dir}'")
        return []

    if "test" not in split:
        csv_files.remove("test.csv")
    if "train" not in split:
        csv_files.remove("train.csv")
    if "validation" not in split:
        csv_files.remove("dev.csv")
    
    # Process each CSV file in the directory
    for file_name in csv_files:
        if "test" in file_name:
            continue
        file_path = os.path.join(data_dir, file_name)
        print(f"Processing file: {file_path}...")
        
        df = pd.read_csv(file_path, encoding='latin1')
        df.dropna(subset=['Word', 'Tag'], inplace=True)

        # --- Group words by sentence ---
        # The 'Sentence #' column marks the start of a new sentence. We forward-fill
        # this value to assign each word to its correct sentence group.
        df['Sentence #'] = df['Sentence #'].ffill()
        grouped_sentences = df.groupby('Sentence #')

        # --- Apply processing logic to each sentence ---
        for sentence_id, sentence_df in grouped_sentences:
            doc_tokens = sentence_df['Word'].tolist()
            doc_tags = sentence_df['Tag'].tolist()

            if not doc_tokens:
                continue

            # Reconstruct text and calculate character offsets for each token
            reconstructed_text = ""
            token_start_offsets = []
            for token in doc_tokens:
                token_start_offsets.append(len(reconstructed_text))
                reconstructed_text += token + " "
            reconstructed_text = reconstructed_text.rstrip()

            # Use seqeval to get token-level entity spans
            extracted_entities = get_entities(doc_tags)
            
            # Convert token-level spans to character-level spans
            doc_entities = []
            for entity_label, start_token_idx, end_token_idx in extracted_entities:
                if entity_label in labels_to_keep:
                    entity_text = " ".join(doc_tokens[start_token_idx : end_token_idx + 1])
                    char_start = token_start_offsets[start_token_idx]
                    last_token_idx = end_token_idx
                    char_end = token_start_offsets[last_token_idx] + len(doc_tokens[last_token_idx])
                    
                    doc_entities.append({
                        "text": entity_text,
                        "label": entity_label,
                        "start": char_start,
                        "end": char_end
                    })
            
            # Assemble the final dictionary for the sentence
            all_processed_sentences.append({
                "text": reconstructed_text,
                "entities": doc_entities
            })

    return all_processed_sentences



def cwed4eta_process_json_file(file_path = CLIRENER_DIR): 
    with open(file_path) as f:
        json_string = json.load(f)
    
    dataset = []
    for task in json_string:
        if not task.get('annotations'):
            continue
        selected_annotations = None
        annotations_by_id = {an['completed_by']: an for an in task.get('annotations', [])}
        
        for annotator_id in CLIRENER_ANNOTATOR_IMPORTANCE:
            if annotator_id in annotations_by_id:
                # If the preferred annotator is found, select their annotations and stop searching
                selected_annotations = annotations_by_id[annotator_id].get('result', [])
                break

        if selected_annotations is None:
            continue
            
        text = task["data"]["sentence"]
        
        entities = []
        # Process the chosen annotations
        for annotation in selected_annotations:
            if annotation.get('type') != 'labels':
                continue
            value = annotation.get('value', {})
            if 'text' in value and 'labels' in value and value['labels']:
                entities.append(
                    {
                        'text': value["text"],
                        'label': value["labels"][0],
                        'start': value["start"],
                        'end': value["end"]
                    }
                )
        dataset.append(
            {
                "text": text,
                "entities": entities
            }
        )
        
    return dataset

# ---------------- PREPROCESSING
def tokenize_text(text):
    """Tokenizes the input text into a list of tokens using a specific regex."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

def convert_to_token_spans(structured_docs_with_char_spans):
    """
    Takes the output from process_bio_documents and converts it to a new format
    with custom tokenization and token-level entity spans.

    Args:
        structured_docs_with_char_spans: A list of dicts, each with 'text'
                                         and 'entities' (with char 'start'/'end').

    Returns:
        A list of dicts, each with 'text', 'tokenized_text', and 'entities'
        (with 'token_start'/'token_end').
    """
    final_data = []

    for doc in structured_docs_with_char_spans:
        text = doc['text']
        
        # 1. Tokenize the text with the custom function
        new_tokens = tokenize_text(text)
        
        # 2. Calculate the character span for each new token
        new_token_char_spans = []
        current_offset = 0
        for token in new_tokens:
            start = text.find(token, current_offset)
            end = start + len(token)
            new_token_char_spans.append((start, end))
            current_offset = end

        # 3. Align original character-level entities with the new token spans
        doc_entities_token_spans = []
        for entity in doc['entities']:
            entity_char_start = entity['start']
            entity_char_end = entity['end']
            
            aligned_token_indices = []
            for i, (token_char_start, token_char_end) in enumerate(new_token_char_spans):
                # Check for overlap between the entity's span and the token's span
                if max(entity_char_start, token_char_start) < min(entity_char_end, token_char_end):
                    aligned_token_indices.append(i)
            
            if aligned_token_indices:
                # The entity's token span is the first and last aligned token index
                token_span_start = aligned_token_indices[0]
                token_span_end = aligned_token_indices[-1]
                
                doc_entities_token_spans.append(
                    [token_span_start, token_span_end, entity["label"]])
        
        # 4. Assemble the final dictionary for the document
        final_data.append({
            # "text": text,
            "tokenized_text": new_tokens,
            "ner": doc_entities_token_spans
        })
        
    return final_data

# ---------------- PROCESSING/LOADING

def load_ibmccner(split = ["train", "validation"]):
    # 1. Load and pre-process the data into a document-wise list of lines
    print(f"Loading '{IBMCCNER_DIR}' with splits: {split}")
    ds = load_dataset(IBMCCNER_DIR) 
    train_documentwise = []
    temp_list = []
    if type(split) == type(["list"]):
        for sp in split:
            for line in ds[sp]["text"]:
                if line.strip().startswith('-DOCSTART-'):
                    if temp_list:
                        train_documentwise.append(temp_list)
                    temp_list = []
                temp_list.append(line)
            if temp_list:
                train_documentwise.append(temp_list)
    else:
        for line in ds[split]["text"]:
            if line.strip().startswith('-DOCSTART-'):
                if temp_list:
                    train_documentwise.append(temp_list)
                temp_list = []
            temp_list.append(line)
        if temp_list:
            train_documentwise.append(temp_list)
    
    print("Converting from BIO tags to char spans.")        
    structured_documents_char_spans = ibmccner_process_bio_documents(
        document_list=train_documentwise,
        labels_to_keep=IBMCCNER_LABELS
    )
    
    print("Converting from char spans to token spans.")
    return convert_to_token_spans(structured_documents_char_spans)

def load_biodivner(split = ["train", "validation"]):
    print(f"Loading '{BIODIVNER_DIR}' with splits: {split}")
    
    if type("string") == type(split):
        split = [split]
        
    print("Converting from BIO tags to char spans.")
    biodivner_structured_data = biodivner_process_bio_documents(
        data_dir=BIODIVNER_DIR, 
        labels_to_keep=BIODIVNER_LABELS, 
        split=split
        )
    
    print("Converting from char spans to token spans.")
    return convert_to_token_spans(biodivner_structured_data)

def load_cwed4eta():
    print(f"Loading '{CLIRENER_DIR}' with no splits.")
    
    print("COnverting to desirable char spans.")
    cwed4eta_structured_data = cwed4eta_process_json_file(CLIRENER_DIR)
    
    print("COnverting from char spans to token spans.")
    return convert_to_token_spans(cwed4eta_structured_data)
    
# ---------------- REFORMATING
def convert_biodivner_to_conll(input_dir: str, output_dir: str):
    """
    Reads all CSV files from an input directory (BiodivNER format), converts them
    to a CoNLL-style format, and saves them to an output directory.

    Args:
        input_dir: The directory containing the source CSV files (e.g., train.csv).
        output_dir: The directory where the converted .txt files will be saved.
    """
    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Find all CSV files to process
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    # 2. Process each CSV file
    for file_name in csv_files:
        input_path = os.path.join(input_dir, file_name)
        # Create a corresponding output filename with a .txt extension
        output_filename = os.path.splitext(file_name)[0] + ".txt"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Converting '{input_path}' to '{output_path}'...")
        
        # Read the CSV file
        df = pd.read_csv(input_path, encoding='latin1')
        df.dropna(subset=['Word', 'Tag'], inplace=True)

        # 3. The key step: Forward-fill the 'Sentence #' to group words correctly
        df['Sentence #'] = df['Sentence #'].ffill()
        
        # 4. Open the output file and write the converted data
        with open(output_path, 'w', encoding='utf-8') as outfile:
            # Group the DataFrame by the sentence identifier
            for _, sentence_df in df.groupby('Sentence #'):
                
                # Write a -DOCSTART- header for each sentence/document block
                outfile.write("-DOCSTART-\tO\n")
                
                # Write each word and tag pair, separated by a tab
                for _, row in sentence_df.iterrows():
                    outfile.write(f"{row['Word']}\t{row['Tag']}\n")
                
                # Write a blank line to separate sentences/documents
                outfile.write("\n")

    print("\nConversion complete.")

def convert_ibmccner_to_conll(document_list):
    """
    Converts a nested list of raw BIO lines from a Hugging Face dataset
    into a single string in the standard CoNLL format.

    Args:
        document_list: A nested list where each inner list contains the raw
                       BIO-formatted lines for one document.

    Returns:
        A single string formatted in the CoNLL BIO style.
    """
    all_conll_parts = []

    # 1. Iterate through each document in the list
    for doc_lines in document_list:
        
        # 2. Add the standard CoNLL document separator
        all_conll_parts.append("-DOCSTART-\tO")
        
        # 3. Process each line within the document
        for line in doc_lines:
            line = line.strip()
            
            # Skip empty lines or the original DOCSTART headers
            if not line or line.startswith('-DOCSTART-'):
                continue
            
            # 4. Split the line into token and tag, and reformat with a tab
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                token, tag = parts
                # The format is 'TOKEN\tTAG'
                all_conll_parts.append(f"{token}\t{tag}")
        
        # 5. Add a blank line to separate documents
        all_conll_parts.append("")
        
    # Join all the processed lines into a single string
    return "\n".join(all_conll_parts)

def convert_predictions_to_conll(predictions):
    """
    Converts a dictionary of sentence predictions with character-level entity spans
    into a single string in CoNLL format with BIO tags.

    Args:
        predictions: A dictionary where keys are sentence IDs and values contain
                     the sentence text and a list of NER predictions with
                     character start/end spans.

    Returns:
        A single string formatted in the CoN-LL BIO style.
    """


    final_conll_parts = []

    # Sort the sentences by their numeric ID to ensure correct order
    sorted_sentence_items = sorted(
        predictions.items(), 
        key=lambda item: int(item[0].split('-')[-1])
    )

    for sentence_id, data in sorted_sentence_items:
        text = data['text']
        ner_predictions = data['ner']
        
        if ner_predictions == "no_entities":
            continue

        # 1. Tokenize the text and calculate the character span for each token
        tokens = text.split()
        token_char_spans = []
        current_offset = 0
        for token in tokens:
            # Find the token's start position, avoiding re-matching the same substring
            start = text.find(token, current_offset)
            end = start + len(token)
            token_char_spans.append((start, end))
            current_offset = end

        # 2. Start with a default list of 'O' tags for every token
        tags = ['O'] * len(tokens)

        # 3. Sort entities by their start position to handle them in order
        sorted_entities = sorted(ner_predictions, key=lambda e: e['start'])

        # 4. Iterate through entities and "paint" the B- and I- tags over the 'O's
        for entity in sorted_entities:
            entity_start_char = entity['start']
            entity_end_char = entity['end']
            label = entity['label']
            
            is_first_token_in_entity = True
            for i, (token_start_char, token_end_char) in enumerate(token_char_spans):
                # Check if the token's span overlaps with the entity's span
                if max(entity_start_char, token_start_char) < min(entity_end_char, token_end_char):
                    # If it's the first token for this entity, tag it as 'B-'
                    if is_first_token_in_entity:
                        tags[i] = f"B-{label}"
                        is_first_token_in_entity = False
                    # Otherwise, tag it as 'I-'
                    else:
                        tags[i] = f"I-{label}"
        
        # 5. Assemble the CoNLL block for this sentence
        final_conll_parts.append(f"-DOCSTART - {sentence_id} -\tO")
        for token, tag in zip(tokens, tags):
            final_conll_parts.append(f"{token}\t{tag}")
        final_conll_parts.append("") # Add a blank line between sentences

    # Join all parts into the final string
    return "\n".join(final_conll_parts)

# ---------------- MISC
def transform_to_ner_format(dataset, labels):
    """
    Transforms the dataset from character-level entity spans to token-level
    BIO tags in the desired format.
    """
    
    # 1. Prepare the Label-to-ID Mapping
    sorted_labels = sorted(list(labels))
    
    # Create the full list of BIO tags
    bio_tags = ["O"] # Start with the "Outside" tag
    for label in sorted_labels:
        bio_tags.append(f"B-{label}")
        bio_tags.append(f"I-{label}")
        
    # Create the mapping from tag string to integer ID
    tag_to_id = {tag: i for i, tag in enumerate(bio_tags)}
    id_for_o = tag_to_id["O"]

    transformed_data = []
    
    # 2. Process Each Data Entry
    for i, entry in enumerate(dataset):
        text = entry['text']
        entities = entry['entities']
        
        if len(text) < 1:
            continue
        
        # Tokenize the text
        tokens = tokenize_text(text)
        
        # Find the character span for each token
        token_spans = []
        current_pos = 0
        for token in tokens:
            # Find the token's start position, searching from the last position
            start = text.find(token, current_pos)
            if start == -1:
                continue # Should not happen with this regex, but good practice
            end = start + len(token)
            token_spans.append({'text': token, 'start': start, 'end': end})
            current_pos = end

        # Initialize NER tags as "O" for all tokens
        ner_tags_str = ["O"] * len(tokens)
        
        # 3. Align Entities with Tokens and Assign Tags
        for entity in entities:
            entity_start = entity['start']
            entity_end = entity['end']
            entity_label = entity['label']
            
            is_first_token = True
            for j, span in enumerate(token_spans):
                # Check if the token is within the entity's span
                # A token is part of the entity if its start and end are within the entity's boundaries
                if span['start'] >= entity_start and span['end'] <= entity_end:
                    if is_first_token:
                        # Assign B-tag for the beginning of an entity
                        ner_tags_str[j] = f"B-{entity_label}"
                        is_first_token = False
                    else:
                        # Assign I-tag for tokens inside an entity
                        ner_tags_str[j] = f"I-{entity_label}"

        # Convert string tags to integer IDs
        ner_tags = [tag_to_id.get(tag, id_for_o) for tag in ner_tags_str]
        
        transformed_data.append({
            'id': str(i),
            'tokens': tokens,
            'ner_tags': ner_tags
        })
        
    return transformed_data, tag_to_id

def ner_dataset_to_hf_format(transformed_dataset, tag_to_id, test_size=0.1, val_size=0.1):
    """
    Takes a list of NER data, performs a multi-label stratified split,
    and returns a Hugging Face DatasetDict.

    Args:
        transformed_dataset (list): The list of {'id', 'tokens', 'ner_tags'} dicts.
        tag_to_id (dict): The mapping from string tags to integer IDs.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float): The proportion of the dataset to include in the validation split.

    Returns:
        datasets.DatasetDict: The final dataset ready for use with Hugging Face.
    """

    # --- 1. Prepare for Stratification ---
    id_to_tag = {i: tag for tag, i in tag_to_id.items()}

    # Extract base entity types per sentence (e.g., "Chemical", "Organism")
    sentence_labels = []
    for entry in transformed_dataset:
        labels = set()
        for tag_id in entry["ner_tags"]:
            tag_name = id_to_tag[tag_id]
            if tag_name != "O":
                base_label = tag_name.split("-")[1]
                labels.add(base_label)
        sentence_labels.append(list(labels))

    # Binarize labels for stratification
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(sentence_labels)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # We'll split on indices, not data itself
    X = np.arange(len(transformed_dataset)).reshape(-1, 1)

    # --- 2. Perform Two-Stage Stratified Split ---

    # Stage 1: Train vs Temp (Val + Test)
    train_X, _, temp_X, temp_y = iterative_train_test_split(
        X, y, test_size=(test_size + val_size)
    )

    # Stage 2: Val vs Test (from Temp)
    relative_test_size = test_size / (test_size + val_size)

    val_X, _, test_X, _ = iterative_train_test_split(
        temp_X, temp_y, test_size=relative_test_size
    )

    # Extract final indices (X holds dataset indices)
    train_indices = train_X.flatten()
    val_indices = val_X.flatten()
    test_indices = test_X.flatten()

    # --- 3. Create Data Subsets ---
    train_data = [transformed_dataset[i] for i in train_indices]
    val_data = [transformed_dataset[i] for i in val_indices]
    test_data = [transformed_dataset[i] for i in test_indices]

    # --- 4. Define Hugging Face Features ---
    bio_tags = sorted(tag_to_id.keys(), key=lambda k: tag_to_id[k])
    features = datasets.Features({
        "id": datasets.Value("string"),
        "tokens": datasets.Sequence(datasets.Value("string")),
        "ner_tags": datasets.Sequence(datasets.ClassLabel(names=bio_tags))
    })

    # --- 5. Assemble the DatasetDict ---
    train_dataset = datasets.Dataset.from_list(train_data, features=features)
    val_dataset = datasets.Dataset.from_list(val_data, features=features)
    test_dataset = datasets.Dataset.from_list(test_data, features=features)

    dataset_dict = datasets.DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    return dataset_dict

def analyze_annotation_data(file_path: str) -> pd.DataFrame:
    """
    Parses a Label Studio JSON export to count total and unique entities per type
    and collects all entity texts for Top-N analysis.

    Args:
        file_path: The path to the Label Studio JSON export file.

    Returns:
        A pandas DataFrame with counts and diversity ratio for each entity type.
        Also returns a dictionary mapping entity types to a list of their texts.
    """
    total_entity_counts = defaultdict(int)
    unique_entities = defaultdict(set)
    all_entity_texts = defaultdict(list) # <-- NEW: To store all entity texts for Top-N

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for task in data:
        if not task.get('annotations'):
            continue
            
        selected_annotations = None
        annotations_by_id = {an['completed_by']: an for an in task.get('annotations', [])}
        
        for annotator_id in CLIRENER_ANNOTATOR_IMPORTANCE:
            if annotator_id in annotations_by_id:
                # If the preferred annotator is found, select their annotations and stop searching
                selected_annotations = annotations_by_id[annotator_id].get('result', [])
                break

        if selected_annotations is None:
            continue    
        
        for annotation in selected_annotations:
            if annotation.get('type') != 'labels':
                continue
            
            value = annotation.get('value', {})
            if 'text' in value and 'labels' in value and value['labels']:
                entity_label = value['labels'][0]
                entity_text = value['text']
                
                total_entity_counts[entity_label] += 1
                unique_entities[entity_label].add(entity_text.lower())
                # NEW: Append the lowercased text for Top-N analysis
                all_entity_texts[entity_label].append(entity_text.lower())

    analysis_data = []
    for label in sorted(total_entity_counts.keys()):
        total_count = total_entity_counts[label]
        unique_count = len(unique_entities[label])
        diversity_ratio = unique_count / total_count if total_count > 0 else 0
        analysis_data.append({
            'EntityType': label,
            'TotalCount': total_count,
            'UniqueCount': unique_count,
            'DiversityRatio': diversity_ratio
        })
    
    df = pd.DataFrame(analysis_data)
    # The second return value is the new dictionary of entity texts
    return df, all_entity_texts
