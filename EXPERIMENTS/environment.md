Command to create a unified SPANBERT/GLiNER environment:

```shell
conda create -n clirener_finetune python=3.10
conda activate clirener_finetune

# GLiNER part (conda installs)
conda install gliner accelerate seqeval datasets
conda install pip

# SpanMarker part (pip installs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install datasets==3.0.0
pip install "transformers<=4.50.0"
pip install span_marker

# Kernel last
conda install -n clirener_finetune ipykernel --update-deps --force-reinstall
conda install matplotlib
conda install scikit-multilearn
conda install seqeval
pip install wandb
pip install nervaluate
```