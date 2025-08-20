import os
import json
import pickle
import argparse
from pathlib import Path
import random
import re
import time
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_from_disk
import pytorch_lightning as pl

from models.transformerpp import ModelArgs, Transformer
from train import GPTTrainingModel
from utils import CHR_DICT, COT_CHR_DICT, GRAPH_MAXCUT_CHR_DICT, SEQ_CHR_DICT, SORTING_CHR_DICT, GRAPH_CHR_DICT

# Import helper functions from recreate_exp
from recreate_exp import (
    extract_dataset_info,
    recreate_dataset,
    create_args_from_config,
    train_from_config_filepath,
)

# We'll need to modify train_from_config_filepath or create a custom training function
# that saves checkpoints at specified steps

# Import activation analysis functions from the existing script
from train_feature_probes_across_compute import (
    extract_relative_activations_and_labels,
    train_and_evaluate,
    run_inference,
    LABELING_FUNCTIONS,
    get_kth_partial_product_carry_function
)

########################################
# Config and Checkpoint Selection
########################################

def collect_run_info(individual_runs_dir):
    """
    Walk through each compute budget subdirectory in individual_runs_dir.
    For each, collect the best run (lowest validation loss) per (compute_budget, (n_layer, n_embd)).
    Returns a dict keyed by (compute_budget, (n_layer, n_embd)) with record details.
    """
    best_runs = {}
    for compute_dir in os.listdir(individual_runs_dir):
        compute_path = os.path.join(individual_runs_dir, compute_dir)
        if not os.path.isdir(compute_path):
            continue
        try:
            compute_budget = int(compute_dir)
        except ValueError:
            print(f"Skipping non-integer compute budget directory: {compute_dir}")
            continue
        for file in os.listdir(compute_path):
            if not file.endswith(".json"):
                continue
            file_path = os.path.join(compute_path, file)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {file_path}")
                continue
            summary = data.get("summary", {})
            config = data.get("config", {})
            val_loss = summary.get("val_loss")
            n_layer = config.get("n_layer")
            n_embd = config.get("n_embd")
            if val_loss is None or n_layer is None or n_embd is None:
                continue
            key = (compute_budget, (n_layer, n_embd))
            if key not in best_runs or val_loss < best_runs[key]["val_loss"]:
                best_runs[key] = {
                    "val_loss": val_loss,
                    "file_path": file_path,
                    "config": data,
                }
    return best_runs

def find_best_config_for_budget(best_runs, target_budget):
    """
    Among all (n_layer, n_embd) for the target_budget, find which has the minimal val_loss.
    Returns (best_n_layer, best_n_embd, best_record).
    """
    best_n_layer = None
    best_n_embd = None
    best_val_loss = float("inf")
    best_record = None
    for (budget, (n_layer, n_embd)), record in best_runs.items():
        if budget == target_budget:
            if record["val_loss"] < best_val_loss:
                best_val_loss = record["val_loss"]
                best_n_layer = n_layer
                best_n_embd = n_embd
                best_record = record
    return best_n_layer, best_n_embd, best_record

def find_matched_configs_across_budgets(best_runs, n_layer, n_embd):
    """
    For the given (n_layer, n_embd), collect the best config (lowest val_loss) for all budgets.
    If n_layer and n_embd are None, then for each budget, select the config with the lowest val_loss.
    Returns a dict: matched[budget] = record.
    """
    matched = {}
    if (n_layer is None) and (n_embd is None):
        for (budget, _), record in best_runs.items():
            if budget in matched:
                if record["val_loss"] < matched[budget]["val_loss"]:
                    matched[budget] = record
            else:
                matched[budget] = record
    else:
        for (budget, (layer, embd)), record in best_runs.items():
            if layer == n_layer and embd == n_embd:
                matched[budget] = record
    return matched

def get_final_checkpoint_step(checkpoint_dir):
    """
    Returns the final step number from the checkpoint directory
    by finding the non-N-Step checkpoint file.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    
    for file in os.listdir(checkpoint_dir):
        if file.startswith("epoch=") and file.endswith(".ckpt"):
            # Extract step number using regex
            match = re.search(r'step=(\d+)', file)
            if match:
                return int(match.group(1))
    
    return None

def calculate_checkpoint_steps_power2(max_steps):
    """
    Calculate steps at which to save checkpoints, using powers of 2.
    Returns a dictionary mapping step number to its power of 2.
    
    Example: If max_steps = 10000
    Returns checkpoints at steps 1, 2, 4, 8, 16, 32, 64, ..., 8192 (assuming 8192 < max_steps < 16384)
    """
    checkpoint_steps = {}
    power = 0
    step = 1
    
    while step <= max_steps:
        checkpoint_steps[step] = power
        power += 1
        step = 2**power
    
    # Also include the final step if it's not already a power of 2
    if max_steps not in checkpoint_steps:
        checkpoint_steps[max_steps] = -1  # -1 indicates this is the max step, not a power of 2
        
    return checkpoint_steps

def get_checkpoint_path(checkpoint_dir, step_num):
    """
    Get the path to the checkpoint for a specific step number.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    
    # Look for a checkpoint with the step number in the filename
    for file in os.listdir(checkpoint_dir):
        if f"step_{step_num}.ckpt" in file:
            return os.path.join(checkpoint_dir, file)
            
    return None

########################################
# Processing Config for Scaling Analysis
########################################

def train_with_power2_checkpoints(config_path, log_dir, checkpoint_steps):
    """
    Train a model using the configuration file and save checkpoints at power-of-2 steps.
    
    Args:
        config_path: Path to the configuration file
        log_dir: Directory where checkpoints should be saved
        checkpoint_steps: Dictionary mapping step numbers to power (of 2)
    
    Returns:
        args: The configuration arguments
    """
    # Extract arguments from the config file
    args = create_args_from_config(config_path)
    
    # Define a custom checkpoint callback to save at specific steps
    class PowerOfTwoCheckpoint(pl.callbacks.ModelCheckpoint):
        def __init__(self, dirpath, checkpoint_steps):
            super().__init__(dirpath=dirpath, save_top_k=-1, verbose=True)
            self.checkpoint_steps = checkpoint_steps
            self.steps_written = set()
            
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = trainer.global_step
            
            # Check if this step is in our list of checkpoint_steps
            if step in self.checkpoint_steps and step not in self.steps_written:
                # For powers of 2, save with power information
                if self.checkpoint_steps[step] >= 0:
                    power = self.checkpoint_steps[step]
                    filepath = os.path.join(self.dirpath, f"power2_{power}_step_{step}.ckpt")
                else:
                    # For max step
                    filepath = os.path.join(self.dirpath, f"final_step_{step}.ckpt")
                
                trainer.save_checkpoint(filepath)
                print(f"Saved checkpoint at step {step} (2^{self.checkpoint_steps[step] if self.checkpoint_steps[step] >= 0 else 'max'})")
                self.steps_written.add(step)
                    
            # Also call the parent method to maintain normal checkpoint behavior
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
    
    # Create the checkpoint directory
    checkpoint_dir = os.path.join(
        log_dir,
        args.task,
        str(args.task_length),
        args.method,
        str(args.compute_budget),
        "lightning_logs",
        "version_0",
        "checkpoints",
    )
    # os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create the custom checkpoint callback
    checkpoint_callback = PowerOfTwoCheckpoint(
        dirpath=checkpoint_dir,
        checkpoint_steps=checkpoint_steps
    )
    
    # Call train_from_config_filepath with our custom callback
    # This assumes train_from_config_filepath accepts a callbacks parameter
    train_from_config_filepath(config_path, log_dir, other_callbacks=[checkpoint_callback])
    
    return args

def process_config_at_step(config_path, log_dir, device, step=None, random_baseline=False):
    """
    Given a config file and a log directory (where checkpoints are saved),
    use create_args_from_config to extract args and dataset info, build the model,
    load the checkpoint for the specified step,
    run inference on validation and test sets,
    and compute activation analysis metrics using each labeling function.

    Returns a dict with identification info and the computed metrics.
    """
    print(f"\n=== Processing config: {config_path} at step {step} ===")
    # args = create_args_from_config(config_path)

    old_args = create_args_from_config(config_path)

    args = create_args_from_config(
        config_path,
        dataset_root_dir='datasets_linear_probe', 
        val_set_size=10000, 
        test_set_size=1000, 
        dataset_seed=random.randint(0, 65536),
        train_filter_set_path=os.path.join(old_args.dataset_dir, 'train_numbers.npy')
    )

    print(
        f"Extracted args: task={args.task}, task_length={args.task_length}, n_layer={args.n_layer}, n_embd={args.n_embd}, compute_budget={args.compute_budget}"
    )

    # Load datasets.
    print("Loading datasets...")
    train_dataset = load_from_disk(args.train_dataset_filepath)
    test_dataset = load_from_disk(args.test_dataset_filepath)

    print(len(train_dataset))
    block_size = len(train_dataset[0]["ids"])

    if args.task in ["addition", "multiplication", "binary_sorting", "parity", "majority", "majority_of_majority", "inner_product_mod2_parity"]:
        if args.method == "normal":
            tokens_dict = CHR_DICT
        elif args.method == "cot":
            tokens_dict = COT_CHR_DICT
        else:
            raise ValueError("Invalid method")
    elif args.task in ["maximum_subarray", "activity_selection", "longest_common_subsequence"]:
        tokens_dict = SEQ_CHR_DICT
    elif args.task == "sorting":
        tokens_dict = SORTING_CHR_DICT
    elif args.task == "graph_maxcut":
        tokens_dict = GRAPH_MAXCUT_CHR_DICT
    elif "graph" in args.task:
        tokens_dict = GRAPH_CHR_DICT
        
    # Build transformer configuration.
    transformerpp_config = ModelArgs(
        max_seq_len=block_size,
        vocab_size=len(tokens_dict.keys()) if tokens_dict else 100,
        n_layers=args.n_layer,
        n_heads=args.n_head,
        dim=args.n_embd,
        max_batch_size=args.test_batch_size,
        multiple_of=2,
        task=args.task,
    )

    base_model = Transformer(transformerpp_config)
    # Create a GPTTrainingModel instance (dataloaders are None since we run inference only).
    pl_model = GPTTrainingModel(
        base_model,
        int(args.max_steps * args.warmup_ratio),
        args.max_steps,
        args.lr,
        args.task_length,
        args.betas,
        args.weight_decay,
        None,
        None,
        None,
        repetitions=1,
        method=args.method,
        log_activations=True,
    )
    pl_model.to(device)

    # Define the checkpoint directory path
    checkpoint_dir = os.path.join(
        log_dir,
        args.task,
        str(args.task_length),
        args.method,
        str(args.compute_budget),
        "lightning_logs",
        "version_0",
        "checkpoints",
    )

    # Load checkpoint if available and not using random baseline
    if not random_baseline:
        if os.path.isdir(checkpoint_dir):
            # Look for a checkpoint for this specific step
            if step is not None:
                checkpoint_path = get_checkpoint_path(checkpoint_dir, step)
                if checkpoint_path:
                    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    pl_model.load_state_dict(state["state_dict"])
                    print(f"Loaded checkpoint for step {step} from {checkpoint_path}")
                else:
                    print(f"No checkpoint found for step {step}; using random weights.")
            else:
                # Default behavior: load the final checkpoint
                ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("final_step_") or f.startswith("epoch=")]
                if ckpt_files:
                    ckpt_filepath = os.path.join(checkpoint_dir, ckpt_files[0])
                    state = torch.load(ckpt_filepath, map_location=device, weights_only=False)
                    pl_model.load_state_dict(state["state_dict"])
                    print(f"Loaded final checkpoint from {ckpt_filepath}")
                else:
                    print("No valid checkpoint file; using random weights.")
        else:
            print("Checkpoint directory not found; using random weights.")
    else:
        print("Random baseline selected: skipping checkpoint loading.")

    # Create DataLoaders.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # Run inference.
    print("Running inference on validation set...")
    train_input_ids, train_activations, train_per_token_losses = run_inference(pl_model, train_loader, device)
    print("Running inference on test set...")
    test_input_ids, test_activations, test_per_token_losses = run_inference(pl_model, test_loader, device)

    train_input_ids = np.array(train_input_ids)
    test_input_ids = np.array(test_input_ids)

    # Loop over each labeling function for the given task.
    all_metrics = {}
    if args.task == "multiplication":
        for idx in [2]: # range(1, args.task_length + 1):
            LABELING_FUNCTIONS["multiplication"][f"carry_{idx}"] = get_kth_partial_product_carry_function(idx)

    if args.task in ["maximum_subarray"]:
        train_input_ids[:, :args.task_length] -= 9
        test_input_ids[:, :args.task_length] -= 9

    if args.task in LABELING_FUNCTIONS:
        for label_name, labeling_func in LABELING_FUNCTIONS[args.task].items():
            
            print(f"\n--- Processing labeling function: {label_name} ---")
            print("Extracting relative activations and labels...")
            val_results = extract_relative_activations_and_labels(
                train_input_ids, train_activations, args.task, label_name, labeling_func
            )
            test_results = extract_relative_activations_and_labels(
                test_input_ids, test_activations, args.task, label_name, labeling_func
            )

            print("Training classifiers and computing metrics...")
            metrics = train_and_evaluate(val_results, test_results)
            all_metrics[label_name] = metrics
    else:
        print(f"No labeling function defined for task {args.task}. Skipping.")
        return None

    # Get the description of what checkpoint was used
    compute_description = "random" if random_baseline else (
        f"step_{step}" if step is not None else "final"
    )

    # Calculate power of 2 if it's a power of 2
    if step is not None and not random_baseline:
        power = None
        if step > 0 and (step & (step - 1) == 0):  # Check if step is a power of 2
            power = int(np.log2(step))
        
    info = {
        "config_path": config_path,
        "compute_budget": args.compute_budget,
        "n_layer": args.n_layer,
        "n_embd": args.n_embd,
        "compute_description": compute_description,
        "step": step,
        "power": power if 'power' in locals() else None,
        "metrics": all_metrics,
        "val_per_token_losses": train_per_token_losses,
        "test_per_token_losses": test_per_token_losses,
        "train_dataset_filepath": args.train_dataset_filepath,
        "test_dataset_fielpath": args.test_dataset_filepath
    }
    return info
    return info

########################################
# Main Routine
########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing experiment results (should have subdir {task_length}/individual_runs)",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory where checkpoints are saved",
    )
    parser.add_argument(
        "--task_length", type=str, required=True, help="Task length to filter for"
    )
    parser.add_argument(
        "--target_budget_exponent",
        type=int,
        required=True,
        help="Target compute budget exponent (e.g., 15 for 10^15)",
    )
    parser.add_argument(
        "--include_random_baseline",
        action="store_true",
        help="If set, will also process metrics for the random baseline",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="If set, will skip the training phase and only run evaluation on existing checkpoints",
    )
    parser.add_argument(
        "--output_pickle",
        type=str,
        default="metrics.pkl",
        help="Output pickle file to save metrics",
    )
    args = parser.parse_args()

    target_budget = 10**args.target_budget_exponent
    individual_runs_dir = os.path.join(
        args.directory, args.task_length, "individual_runs"
    )
    if not os.path.isdir(individual_runs_dir):
        raise ValueError(f"Individual runs directory not found: {individual_runs_dir}")

    # 1) Build a dictionary of best runs for each (compute_budget, (n_layer, n_embd))
    best_runs = collect_run_info(individual_runs_dir)

    # 2) Find which (n_layer, n_embd) is best for the target_budget
    best_n_layer, best_n_embd, best_record = find_best_config_for_budget(
        best_runs, target_budget
    )
    if best_record is None:
        print(f"No valid runs found for compute_budget={target_budget}. Exiting.")
        return

    print(f"\nFor target budget={target_budget}, the best config is:")
    print(f"  * n_layer={best_n_layer}")
    print(f"  * n_embd={best_n_embd}")
    print(f"  * val_loss={best_record['val_loss']}")
    print(f"  * file_path={best_record['file_path']}")

    # Get the model configuration and max steps
    model_args = create_args_from_config(best_record['file_path'])
    max_steps = model_args.max_steps
    print(f"Max training steps for target budget: {max_steps}")

    # Calculate the steps corresponding to powers of 2
    checkpoint_steps = calculate_checkpoint_steps_power2(max_steps)
    
    # Display the calculated checkpoint steps
    print("\nCheckpoint steps at powers of 2:")
    for step, power in sorted(checkpoint_steps.items()):
        if power >= 0:
            print(f"  Step {step} = 2^{power}")
        else:
            print(f"  Step {step} = final")

    # 3) Train the model and save checkpoints at the specified steps
    if not args.skip_training:
        print("\n--- Training model and saving checkpoints at powers of 2 steps ---")
        train_with_power2_checkpoints(
            best_record['file_path'],
            args.ckpt_dir,
            checkpoint_steps
        )
    else:
        print("\n--- Skipping training phase (--skip_training flag set) ---")

    # 4) Process activation analysis at each checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overall_metrics = {}

    # Process the final step
    print(f"\n--- Processing activation analysis for final step (max_steps={max_steps}) ---")
    info = process_config_at_step(
        best_record["file_path"], 
        args.ckpt_dir, 
        device, 
        max_steps
    )
    if info is not None:
        overall_metrics[max_steps] = info

    # Process each power of 2 step
    for step, power in sorted(checkpoint_steps.items()):
        if step != max_steps:  # Skip the final step (already processed)
            print(f"\n--- Processing activation analysis for step {step} (2^{power}) ---")
            info = process_config_at_step(
                best_record["file_path"], 
                args.ckpt_dir, 
                device, 
                step
            )
            if info is not None:
                overall_metrics[step] = info

    # Optionally process random baseline
    if args.include_random_baseline:
        print("\n--- Processing activation analysis for random baseline ---")
        info = process_config_at_step(
            best_record["file_path"], 
            args.ckpt_dir, 
            device, 
            None,  # No specific step for random baseline
            random_baseline=True
        )
        if info is not None:
            overall_metrics["random"] = info

    # Save all metrics
    with open(args.output_pickle, "wb") as f:
        pickle.dump(overall_metrics, f)
    print(f"\nSaved overall metrics to {args.output_pickle}")

if __name__ == "__main__":
    main()