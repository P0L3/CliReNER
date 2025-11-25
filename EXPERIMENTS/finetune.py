import argparse

parser = argparse.ArgumentParser(
                    prog='ModelFineTuning',
                    description='Program uses HuggingFace Dataset to fine-tune NER models.',
                    epilog='...')

parser.add_argument("--model_type", type=str)

args = parser.parse_args()

###