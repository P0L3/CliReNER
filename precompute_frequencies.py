import json
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets

def main():
    dataset_name = "P0L3/CliReNER_v_1_1_28_SILVER"
    output_file = "label_frequencies.json"
    
    print(f"Loading dataset: {dataset_name}...")
    hf_dataset = load_dataset(dataset_name)
    
    # Get Label Names
    available_split = list(hf_dataset.keys())[0]
    tags_feature = hf_dataset[available_split].features["ner_tags"].feature
    label_names = tags_feature.names

    # Combine all datasets to get the global distribution
    datasets_list =[hf_dataset[split] for split in hf_dataset.keys()]
    combined_dataset = concatenate_datasets(datasets_list)
    
    class_counts = defaultdict(int)
    
    print("Calculating frequencies...")
    for row in combined_dataset:
        for tag_id in row['ner_tags']:
            if tag_id < 0 or tag_id >= len(label_names):
                continue
            tag_name = label_names[tag_id]
            
            # We only count "B-" to get the number of entities (matching your stats script)
            if tag_name.startswith("B-"):
                class_type = tag_name[2:]
                class_counts[class_type] += 1
                
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(class_counts, f, indent=4)
        
    print(f"Successfully saved {len(class_counts)} class frequencies to {output_file}!")

if __name__ == "__main__":
    main()