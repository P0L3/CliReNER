# CliReNER

## Steps Manual
1. Using [ed4re_50_papers_sample](ed4re_50_papers_sample.ipynb) we explore the full data and choose appropriate journal articles for annotation.
2. After this, using [gliner_try](gliner_try.ipynb) we use GLiNER model to perform zero-shot "annotation" of the data (which will be used as entity suggestions).
3. Results from GLiNER are then converted to labelstudio format using [gliner_results2labelstudio](gliner_results2labelstudio.ipynb). This includes merging with existing annotations (categories mi9ght require an update) and also providing ready annotator datasets (without annotations) based on label set and occurance requirements.
4. Annotations results exported from Lable Studio can be analyzed using [labelstudio_annots_analysis](labelstudio_annots_analysis.ipynb). Output examples are in [this directory](PLOTS/ANNOTATION/).
5. Using [labelstudio_parsing](labelstudio_parsing.ipynb) a HuggingFace dataset is created and pushed to the hub (Note: privacy settings are edited manually). 

## Experiment Steps
1. Data from Step 3 is used for creation of two HuggingFace Datasets. (Results from GLiNER are then converted to labelstudio format using [gliner_results2labelstudio](gliner_results2labelstudio.ipynb). ...)
2. With [create_hf_dataset](EXPERIMENTS/create_hf_dataset.py) the two HF Datasets are created. For exact commands check [commands](EXPERIMENTS/commands.md).
3. Dataset statistics are calculated with [calculate_hf_dataset_stats](EXPERIMENTS/calculate_hf_dataset_stats.py).Plots are available in [DATASET_STATS folder](EXPERIMENTS/DATASET_STATS).
4. For model fine-tuning ([GLiNER](https://github.com/urchade/GLiNER) and [SPANMARKER](https://github.com/tomaarsen/SpanMarkerNER)) [finetune](EXPERIMENTS/finetune.py) and [evaluate](EXPERIMENTS/evaluate.py) scripts are used. For the full experiments with multiple seeds we use [finetune-evaluate pipeline](EXPERIMENTS/finetune_evaluate_pipeline.py) with specific parameters ([gliner_config](EXPERIMENTS/gliner_config.json) and [spanmarker_config](EXPERIMENTS/spanmarker_config.json)) and arguments ([gliner](EXPERIMENTS/run_seeds_gliner.sh) and [spanmarker](EXPERIMENTS/run_seeds_spanmarker.sh) shell scripts) available. The dataset used for fine-tuning and initial evaluation is [P0L3/CliReNER_v_1_1_28_SILVER](https://huggingface.co/datasets/P0L3/CliReNER_v_1_1_28_SILVER).
5. Evaluation on GOLD dataset is performed using [evaluate_gold](EXPERIMENTS/evaluate_gold.py).


## To Do
- [x] <del>Unify GLiNER and SPANMARKER finetuning procedure.</del>
  - [x] <del>Unify data source using HF dataset (GLiNER currently work differently).</del>
  - [x] <del>Add SEED option to finetuning (reproducability).</del>
  - [x] <del>Add WANDB wrapper for monitoring.</del>
- [ ] Update Dataset card for CLIRENER_V_1_0_28
- [x] <del>Add useful outputs for WANDB visualization</del>
- [ ] Correct experiment configs for SILVER experiment
