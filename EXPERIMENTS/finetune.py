import argparse

parser = argparse.ArgumentParser(
                    prog='ModelFineTuning',
                    description='Program uses HuggingFace Dataset to fine-tune NER models.',
                    epilog='...')

parser.add_argument("--model_type", type=str)

args = parser.parse_args()

if args.model_type == "GLINER":
    # import json
    # import random
    # from datasets import load_dataset
    # from seqeval.metrics.sequence_labeling import get_entities
    # import re
    # from collections import Counter
    # # import matplotlib.pyplot as plt
    # import pandas as pd
    # from typing import List, Dict

    # from dataset_processing import *
    print("GLINER")
    
elif args.model_type == "SPANMARKER":
    print("SPANMARKER")