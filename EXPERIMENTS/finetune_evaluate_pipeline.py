import argparse
import subprocess
import sys
import re
import wandb # <--- NEEDED to generate the ID
from pathlib import Path
from EXPERIMENTS.finetune import shorten_name, get_output_dir

def get_expected_model_path(base_dir, model_type, model_id, dataset_id, seed=None):
    return Path(get_output_dir(base_dir, model_type, model_id, dataset_id, seed) / "checkpoint-final")

def run_command(command, description):
    print(f"\n{'='*10} Starting {description} {'='*10}")
    print(f"Executing: {' '.join(command)}\n")
    try:
        subprocess.check_call(command)
        print(f"\n{'='*10} Finished {description} Successfully {'='*10}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n!!!!!! {description} FAILED with exit code {e.returncode} !!!!!!")
        sys.exit(e.returncode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["GLINER", "SPANMARKER"])
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    # 1. Generate the Run ID here (The Source of Truth)
    run_id = wandb.util.generate_id()
    print(f"--- Generated WandB Run ID: {run_id} ---")

    run_name_with_seed = f"{args.wandb_name}_seed{args.seed}"
    
    # 2. Training
    train_cmd = [
        sys.executable, "-m", "EXPERIMENTS.finetune",
        "--model_type", args.model_type,
        "--dataset_id", args.dataset_id,
        "--model_id", args.model_id,
        "--config_path", args.config_path,
        "--wandb_project", args.wandb_project,
        "--wandb_name", run_name_with_seed,
        "--wandb_run_id", run_id,
        "--seed", str(args.seed)
    ]
    if args.wandb_entity:
        train_cmd.extend(["--wandb_entity", args.wandb_entity])

    run_command(train_cmd, "TRAINING")

    # 3. Path Calculation
    saved_model_path = get_expected_model_path("EXPERIMENTS/models", args.model_type, args.model_id, args.dataset_id, args.seed)
    
    if not saved_model_path.exists():
        print(f"Error: Expected model path does not exist: {saved_model_path}")
        sys.exit(1)

    # 4. Evaluation
    eval_cmd = [
        sys.executable, "-m", "EXPERIMENTS.evaluate",
        "--model_type", args.model_type,
        "--dataset_id", args.dataset_id,
        "--model_path", str(saved_model_path),
        "--wandb_project", args.wandb_project,
        "--wandb_run_name", run_name_with_seed,
        "--wandb_run_id", run_id  # <--- PASSING THE SAME ID
    ]
    if args.wandb_entity:
        eval_cmd.extend(["--wandb_entity", args.wandb_entity])

    run_command(eval_cmd, "EVALUATION")