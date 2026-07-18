import time
import torch
from datasets import load_dataset, concatenate_datasets
from gliner import GLiNER
from dataset_processing import CLIRENER_LABELS_V1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load a realistic batch of text (from your Gold dataset)
    print("Loading text samples...")
    ds = load_dataset("P0L3/CliReNER_v_1_1_28_GOLD")
    merged = concatenate_datasets([ds[s] for s in ds.keys()])
    texts = [row["text"] for row in merged]
    
    # To get a stable benchmark, duplicate the texts so we have ~2000 sentences
    while len(texts) < 2000:
        texts.extend(texts)
    texts = texts[:2000] 
    
    # 2. Load Model
    print("Loading GLiNER-Medium...")
    model = GLiNER.from_pretrained("P0L3/CliReNER-gliner_medium-v2.5", load_tokenizer=True)
    model.to(device)
    
    labels = list(CLIRENER_LABELS_V1)
    
    # 3. Warmup (Run a few sentences so CUDA initializes)
    print("Warming up GPU...")
    for t in texts[:10]:
        _ = model.predict_entities(t, labels)
    
    # 4. Benchmark (Sequential, 1-by-1)
    print(f"Benchmarking throughput on {len(texts)} sentences (Per-Sentence Inference)...")
    start_time = time.time()
    
    for t in texts:
        _ = model.predict_entities(t, labels)
        
    end_time = time.time()
    
    total_time = end_time - start_time
    sentences_per_second = len(texts) / total_time
    
    print("\n" + "="*50)
    print("🚀 GLiNER INFERENCE BENCHMARK (UNBATCHED)")
    print("="*50)
    print(f"Total time for {len(texts)} sentences: {total_time:.2f} seconds")
    print(f"Throughput: {sentences_per_second:.2f} sentences/second")
    
    # 5. Extrapolate to 28.4 Million
    total_seconds_required = 28_400_000 / sentences_per_second
    gpu_hours = total_seconds_required / 3600
    
    print(f"\nExtrapolation for 100,000 papers (28.4M sentences):")
    print(f"Required GPU Time: {gpu_hours:.2f} Hours")
    print("="*50)

if __name__ == "__main__":
    main()