# CliReNER

## Steps
1. Using [ed4re_50_papers_sample](ed4re_50_papers_sample.ipynb) we explore the full data and choose appropriate journal articles for annotation.
2. After this, using [gliner_try](gliner_try.ipynb) we use GLiNER model to perform zero-shot "annotation" of the data (which will be used as entity suggestions).
3. Results from GLiNER are then converted to labelstudio format using [gliner_results2labelstudio](gliner_results2labelstudio.ipynb). This includes merging with existing annotations (categories mi9ght require an update) and also providing ready annotator datasets (without annotations) based on label set and occurance requirements.
