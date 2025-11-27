# CliReNER

## Steps
1. Using [ed4re_50_papers_sample](ed4re_50_papers_sample.ipynb) we explore the full data and choose appropriate journal articles for annotation.
2. After this, using [gliner_try](gliner_try.ipynb) we use GLiNER model to perform zero-shot "annotation" of the data (which will be used as entity suggestions).
3. Results from GLiNER are then converted to labelstudio format using [gliner_results2labelstudio](gliner_results2labelstudio.ipynb). This includes merging with existing annotations (categories mi9ght require an update) and also providing ready annotator datasets (without annotations) based on label set and occurance requirements.
4. Annotations results exported from Lable Studio can be analyzed using [labelstudio_annots_analysis](labelstudio_annots_analysis.ipynb). Output examples are in [this directory](PLOTS/ANNOTATION/).
5. Using [labelstudio_parsing](labelstudio_parsing.ipynb) a HuggingFace dataset is created and pushed to the hub (Note: privacy settings are edited manually).
6. 


## To Do
- [x] <del>Unify GLiNER and SPANMARKER finetuning procedure.</del>
  - [x] <del>Unify data source using HF dataset (GLiNER currently work differently).</del>
  - [x] <del>Add SEED option to finetuning (reproducability).</del>
  - [x] <del>Add WANDB wrapper for monitoring.</del>
- [ ] Update Dataset card for CLIRENER_V_1_0_28
- [x] <del>Add useful outputs for WANDB visualization</del>
- [ ] Correct experiment configs for SILVER experiment
