Command to create a unified SPANBERT/GLiNER environment:

```shell
conda create -n clirener_finetune python=3.10
conda activate clirener_finetune

# GLiNER part (conda installs)
conda install gliner accelerate seqeval datasets -y
conda install pip -y

# SpanMarker part (pip installs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install datasets==3.0.0
pip install "transformers<=4.50.0"
pip install span_marker

# Kernel last
conda install -n clirener_finetune ipykernel --update-deps --force-reinstall -y
conda install matplotlib -y
conda install scikit-multilearn -y
conda install seqeval -y
pip install wandb 
pip install nervaluate 
pip install multiset-multicover
pip install krippendorff

pip install spacy==3.7.5
```