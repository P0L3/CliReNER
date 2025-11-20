from dataset_processing import cwed4eta_process_json_file, transform_to_ner_format, CLIRENER_LABELS_V1, ner_dataset_to_hf_format
import argparse
import pathlib

parser = argparse.ArgumentParser(
                    prog='LSdata2HF',
                    description='Program transforms Label Studio data to BIO format and uploads it to HuggingFace.',
                    epilog='...')
parser.add_argument(
    "--lsfile_path", type=pathlib.Path
)
args = parser.parse_args()

print("Loading data from: ", args.lsfile_path)
data = cwed4eta_process_json_file(args.lsfile_path)

print("Transforming data to BIO tags ...")
transformed_dataset, tag_map = transform_to_ner_format(data, CLIRENER_LABELS_V1)

print("Transforming data to HF Dataset ...")
hf_dataset = ner_dataset_to_hf_format(transformed_dataset, tag_map)

print("Dataset info:")
print(hf_dataset)