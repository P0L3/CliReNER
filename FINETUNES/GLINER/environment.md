# 1. Create and activate environment
conda create -n clirener_finetune_gliner python=3.10
conda activate clirener_finetune_gliner

# 2. GLiNER dependencies (Conda)
conda install gliner accelerate seqeval datasets -y
conda install pip -y
pip install gliner2

# 3. SpanMarker dependencies (Pip)
# Note: Specific versions are required for CUDA/Torch compatibility
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install datasets==3.0.0
pip install "transformers<=4.50.0"
# pip install span_marker

# 4. Utilities & Kernels
conda install -n clirener_finetune_gliner ipykernel --update-deps --force-reinstall -y
# conda install matplotlib scikit-multilearn seqeval -y
# pip install wandb nervaluate multiset-multicover spacy==3.7.5
pip install spacy scispacy matplotlib scikit-multilearn peft wandb nervaluate