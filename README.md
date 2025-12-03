# CliReNER
**Cli**mate **Re**search **N**amed **E**ntity **R**ecognition

CliReNER is a comprehensive framework designed to extract structured information from climate research literature. It provides a full pipeline from data selection and zero-shot pre-annotation (using GLiNER) to human verification (Label Studio) and model fine-tuning (GLiNER & SpanMarker).

## 🛠 Installation & Environment

To ensure compatibility between GLiNER and SpanMarker, a specific environment setup is required. You can find detailed notes in [`EXPERIMENTS/environment.md`](EXPERIMENTS/environment.md).

**Quick Setup:**

```bash
# 1. Create and activate environment
conda create -n clirener_finetune python=3.10
conda activate clirener_finetune

# 2. GLiNER dependencies (Conda)
conda install gliner accelerate seqeval datasets -y
conda install pip -y

# 3. SpanMarker dependencies (Pip)
# Note: Specific versions are required for CUDA/Torch compatibility
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install datasets==3.0.0
pip install "transformers<=4.50.0"
pip install span_marker

# 4. Utilities & Kernels
conda install -n clirener_finetune ipykernel --update-deps --force-reinstall -y
conda install matplotlib scikit-multilearn seqeval -y
pip install wandb nervaluate multiset-multicover spacy==3.7.5
```

## 📂 Project Structure

Below is an overview of the key directories and scripts in the project:

```text
.
├── DATA/                       # Raw and processed datasets (ClimateIE, LabelStudio exports, etc.)
├── EXPERIMENTS/                # Core scripts for training and evaluation
│   ├── DATASET_STATS/          # Output plots for dataset statistics
│   ├── commands.md             # Detailed command reference
│   ├── create_hf_dataset.py    # Script to build HuggingFace datasets
│   ├── finetune.py             # Single-run fine-tuning script
│   ├── finetune_evaluate_pipeline.py  # Multi-seed full experiment pipeline
│   ├── gliner_config.json      # Hyperparameters for GLiNER
│   ├── run_seeds_gliner.sh     # Shell script to run 5 seed fine-tuning experiments - GLiNER
│   ├── run_seeds_spanmarker.sh # Shell script to run 5 seed fine-tuning experiments - SPANMARKER
│   └── spanmarker_config.json  # Hyperparameters for SpanMarker
├── FINETUNES/                  # Previous model fine-tuning experiments
├── PLOTS/                      # Visualization outputs (Annotation agreement, Dataset overlaps)
├── RESULTS/                    # Raw JSON results from model inferences
├── GUIDELINES/                 # Annotation guidelines (PDF)
├── *.ipynb                     # Jupyter notebooks for data exploration and pre-processing
└── README.md
```

## 1. Data Preparation & Annotation Pipeline

This pipeline covers the lifecycle of data from raw text to HuggingFace dataset.

1.  **Data Selection:**
    *   Use [`ed4re_50_papers_sample.ipynb`](ed4re_50_papers_sample.ipynb) to explore the `ed4re` corpus and select representative journal articles for annotation.
2.  **Zero-Shot Pre-annotation:**
    *   Run [`gliner_try.ipynb`](gliner_try.ipynb) to generate initial entity suggestions using the GLiNER model.
3.  **Label Studio Conversion:**
    *   Use [`gliner_results2labelstudio.ipynb`](gliner_results2labelstudio.ipynb) to convert GLiNER results into Label Studio format.
    *   *Note:* This merges existing annotations and creates ready-to-import tasks, filtering based on label occurrence requirements.
4.  **Analysis & Export:**
    *   Analyze annotation quality and agreement using [`labelstudio_annots_analysis.ipynb`](labelstudio_annots_analysis.ipynb). Output examples are available in [`PLOTS/ANNOTATION/`](PLOTS/ANNOTATION/).
    *   Finally, use [`labelstudio_parsing.ipynb`](labelstudio_parsing.ipynb) to parse the final Label Studio export, create a HuggingFace dataset, and push it to the Hub (privacy settings are handled manually).

## 2. Experimentation Framework

This section details how to create Silver/Gold datasets, fine-tune models, and evaluate performance.

### Dataset Management
*   **Creation:** Use [`EXPERIMENTS/create_hf_dataset.py`](EXPERIMENTS/create_hf_dataset.py) to generate the training datasets. Refer to [`EXPERIMENTS/commands.md`](EXPERIMENTS/commands.md) for the exact CLI arguments. *Note: This relies on outputs from [`gliner_results2labelstudio.ipynb`](gliner_results2labelstudio.ipynb).*
*   **Statistics:** Calculate and visualize dataset statistics using [`EXPERIMENTS/calculate_hf_dataset_stats.py`](EXPERIMENTS/calculate_hf_dataset_stats.py). Plots are saved to [`EXPERIMENTS/DATASET_STATS`](EXPERIMENTS/DATASET_STATS).

### Model Training (Fine-Tuning)
We support fine-tuning for both [GLiNER](https://github.com/urchade/GLiNER) and [SPANMARKER](https://github.com/tomaarsen/SpanMarkerNER). The primary training dataset is [P0L3/CliReNER_v_1_1_28_SILVER](https://huggingface.co/datasets/P0L3/CliReNER_v_1_1_28_SILVER).

*   **Single Run:** Use [`finetune.py`](EXPERIMENTS/finetune.py) and [`evaluate.py`](EXPERIMENTS/evaluate.py) for quick tests.
*   **Full Pipeline (Multiple Seeds):** Use [`finetune_evaluate_pipeline.py`](EXPERIMENTS/finetune_evaluate_pipeline.py) for robust experiments.
    *   **Configs:** [`gliner_config.json`](EXPERIMENTS/gliner_config.json) / [`spanmarker_config.json`](EXPERIMENTS/spanmarker_config.json)
    *   **Execution:**
        *   GLiNER: `bash EXPERIMENTS/run_seeds_gliner.sh`
        *   SpanMarker: `bash EXPERIMENTS/run_seeds_spanmarker.sh`

### Evaluation
*   **Gold Standard:** Perform final evaluation on the GOLD dataset using [`EXPERIMENTS/evaluate_gold.py`](EXPERIMENTS/evaluate_gold.py).

## ✅ Roadmap / To Do

- [x] ~~Unify GLiNER and SPANMARKER finetuning procedure.~~
  - [x] ~~Unify data source using HF dataset.~~
  - [x] ~~Add SEED option to finetuning (reproducibility).~~
  - [x] ~~Add WANDB wrapper for monitoring.~~
- [x] ~~Add useful outputs for WANDB visualization~~
- [ ] Update Dataset card for `CLIRENER_V_1_0_28`
- [ ] Correct experiment configs for SILVER experiment