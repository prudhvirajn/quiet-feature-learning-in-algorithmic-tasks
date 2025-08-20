import os
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from scipy import stats

# Import from compute_intermediate_accuracies
from train_feature_probes_across_compute import (
    collect_run_info,
    find_best_config_for_budget,
    find_matched_configs_across_budgets,
    process_config
)

# Import helper functions from recreate_exp
from recreate_exp import (
    extract_dataset_info,
    recreate_dataset,
    create_args_from_config,
    train_from_config_filepath,
)

from models.transformerpp_concept_ablations import ModelArgs, Transformer
from train import GPTTrainingModel
from utils import CHR_DICT, COT_CHR_DICT, SORTING_CHR_DICT, GRAPH_CHR_DICT, SEQ_CHR_DICT, GRAPH_MAXCUT_CHR_DICT
from datasets import load_from_disk
from torch.utils.data import DataLoader

def load_metrics(pickle_file):
    """Load the metrics from the pickle file."""
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

def find_best_modules(metrics, task, metric_name, specific_label_name=None, target_budget_exp=None, topk=1):
    """
    For each position, find the top-k modules with the lowest loss for the specified metric.
    
    Args:
        metrics: Dictionary of metrics
        task: Task name
        metric_name: Metric to optimize (e.g., test_log_loss, test_acc)
        specific_label_name: If provided, only consider this specific labeling function
        target_budget_exp: If provided, use this budget exponent; otherwise use the highest available
        topk: Number of top modules to return for each position (default: 1)
    
    Returns:
        best_modules: Dictionary mapping positions to lists of (layer_idx, module, coefficients, loss) tuples
        max_position: Maximum position found in metrics (useful for determining seq_len)
    """
    best_modules = {}
    max_position = 0
    
    # Find the target budget metrics
    if target_budget_exp is not None:
        # Use the specified budget
        target_budget = 10**target_budget_exp
        if str(target_budget) not in metrics:
            raise ValueError(f"Specified budget {target_budget} not found in metrics.")
    else:
        # Find the highest non-random budget
        budget_keys = [b for b in metrics.keys() if 'random' not in b]
        if not budget_keys:
            raise ValueError("No non-random budget keys found in metrics.")
        target_budget = max([int(b) for b in budget_keys])
    
    budget_metrics = metrics[str(target_budget)]
    
    print(f"Analyzing metrics for budget: {target_budget}")
    
    # Iterate through all labeling functions and their metrics
    for label_name, label_metrics in budget_metrics['metrics'].items():
        if specific_label_name and label_name != specific_label_name:
            continue
            
        print(f"Processing labeling function: {label_name}")
        
        # Find all positions across all modules
        all_positions = set()
        for layer_idx in label_metrics.keys():
            for module_name in label_metrics[layer_idx].keys():
                if 'attention_scores' in module_name:
                    continue
                if metric_name in label_metrics[layer_idx][module_name]:
                    metric_values = label_metrics[layer_idx][module_name][metric_name]
                    if isinstance(metric_values, np.ndarray):
                        positions = range(len(metric_values))
                        all_positions.update(positions)
                        # Track maximum position
                        if positions:
                            max_position = max(max_position, max(positions))
        
        print(f"Found {len(all_positions)} positions for {label_name}")
        
        # For each position, find the top-k modules
        for position in all_positions:
            # For storing all candidate modules for this position
            candidates = []
            
            for layer_idx in label_metrics.keys():
                for module_name in label_metrics[layer_idx].keys():
                    if 'attention_scores' in module_name:
                        continue
                    if metric_name in label_metrics[layer_idx][module_name]:
                        metric_values = label_metrics[layer_idx][module_name][metric_name]
                        if isinstance(metric_values, np.ndarray) and position < len(metric_values):
                            loss = metric_values[position]
                            # if position == 0:
                            #     print(loss)
                            if not np.isnan(loss):
                                # Get the corresponding coefficient
                                if 'coef' in label_metrics[layer_idx][module_name] and position < len(label_metrics[layer_idx][module_name]['coef']):
                                    coef = label_metrics[layer_idx][module_name]['coef'][position]
                                    intercept = label_metrics[layer_idx][module_name]['intercept'][position] if 'intercept' in label_metrics[layer_idx][module_name] else 0

                                    if len(coef.shape) < 2: coef = coef[np.newaxis, ...]
                                    if len(intercept.shape) < 1: intercept = intercept[np.newaxis, ...]

                                    candidates.append((int(layer_idx), module_name, coef, intercept, loss))
            
            # Sort candidates based on loss
            if "acc" in metric_name:
                # For accuracy metrics, higher is better
                candidates.sort(key=lambda x: x[4], reverse=True)
            else:
                # For loss metrics, lower is better
                candidates.sort(key=lambda x: x[4])
            
            # Take the top-k
            topk_candidates = candidates[:topk]
            
            if topk_candidates:
                key = f"{label_name}_{position}"
                best_modules[key] = topk_candidates
    
    total_modules = sum(len(candidates) for candidates in best_modules.values())
    print(f"Found {total_modules} modules across {len(best_modules)} positions (top-{topk} per position)")
    print(f"Maximum position found in metrics: {max_position}")
    return best_modules, max_position

def map_module_name(module_name):
    """Map the module name from metrics to the expected projection name."""
    if 'attention' in module_name and 'scores' not in module_name:
        return 'attention'
    elif 'ffn' in module_name:
        return 'ffn'
    elif 'attn_residual' in module_name:
        return 'attn_residual'
    elif 'residual' in module_name:
        return 'residual'
    else:
        return module_name

def create_concept_vectors(best_modules, n_embd, device, seq_len=64, start_index=0, end_index=64, pool_positions=False):
    """
    Create a dictionary of concept vectors from the best modules.
    Each module gets a single tensor for weights and biases that spans the sequence length.
    
    Returns a nested dictionary with the structure:
    {layer_idx: {module: {'w': tensor(seq_len, n_embd), 'b': tensor(seq_len)}}}
    
    Args:
        best_modules: Dictionary mapping positions to lists of (layer_idx, module, coefficients) tuples
        n_embd: Embedding dimension
        device: PyTorch device
        seq_len: Sequence length for the concept vectors
    """
    concept_vectors = {}
    module_stats = {}
    
    # First, gather statistics for reporting
    for key, candidates in best_modules.items():
        for layer_idx, module_name, _, _, loss in candidates:
            mapped_module = map_module_name(module_name)
            module_key = f"{layer_idx}_{mapped_module}"
            
            if module_key not in module_stats:
                module_stats[module_key] = []
            
            module_stats[module_key].append((key, loss))

    # Initialize tensors for all modules and positions
    for key, candidates in best_modules.items():
        for layer_idx, module_name, coef, intercept, _ in candidates:
            mapped_module = map_module_name(module_name)
            
            # Initialize layer in concept_vectors if needed
            if layer_idx not in concept_vectors:
                concept_vectors[layer_idx] = {}
            
            # Initialize module tensors if needed
            if mapped_module not in concept_vectors[layer_idx]:
                concept_vectors[layer_idx][mapped_module] = {
                    'w': torch.zeros(seq_len, *coef.shape, device=device),
                    'b': torch.zeros(seq_len, *intercept.shape, device=device),
                    # 'w_norm': torch.zeros(seq_len, coef.shape[0], device=device),
                }
    
    # Fill in specific positions with the coefficients
    # Process all modules for each position
    for key, candidates in best_modules.items():
        if pool_positions:
            position = 0
        else:
            position = int(key.split('_')[-1])
        
        for layer_idx, module_name, coef, intercept, _ in candidates:
            mapped_module = map_module_name(module_name)
            
            # Skip if module not in expected set or position out of range
            if mapped_module not in concept_vectors.get(layer_idx, {}) or position >= seq_len:
                continue
            
            # Process coefficients
            if isinstance(coef, np.ndarray):
                # For multi-class classification or multi-output regression

                # if coef.ndim > 1:
                #     if coef.shape[0] > 1:  # Multiple classes/outputs
                #         # Average across all classes/outputs
                #         w = torch.tensor(np.mean(coef, axis=0), dtype=torch.float32, device=device)
                #     else:  # Single class but reshaped
                #         w = torch.tensor(coef.reshape(-1), dtype=torch.float32, device=device)
                #         b = torch.tensor(intercept, dtype=torch.float32, device=device)
                # else:  # Binary classification or single-output regression
                w = torch.tensor(coef, dtype=torch.float32, device=device)
                b = torch.tensor(intercept, dtype=torch.float32, device=device)

                # Ensure vector has correct dimension
                # if w.shape[0] != n_embd:
                #     print(f"Warning: Weight vector for {key} has wrong dimension: {w.shape[0]} vs expected {n_embd}")
                #     continue
                    
                # Normalize the weight vector

                w_norm = torch.norm(w, p=2, dim=-1)

                w = torch.einsum('sd,s->sd', w, 1/w_norm)
                b = torch.einsum('s,s->s', b, 1/w_norm)
                
                if pool_positions:
                    concept_vectors[layer_idx][mapped_module]['w'][start_index : end_index] = w
                    concept_vectors[layer_idx][mapped_module]['b'][start_index : end_index] = b
                else:
                    concept_vectors[layer_idx][mapped_module]['w'][start_index + position] = w
                    concept_vectors[layer_idx][mapped_module]['b'][start_index + position] = b
    
    # Print statistics on selected modules
    print("\nSelected modules for concept vectors:")
    for module_key, stats in module_stats.items():
        layer_idx, module_name = module_key.split('_', 1)
        print(f"Layer {layer_idx}, Module {module_name}: {len(stats)} positions")
        for key, value in sorted(stats[:5]):  # Print first 5 as examples
            pos = int(key.split('_')[-1])
            print(f"  - Position {pos}: {value:.4f}")
        if len(stats) > 5:
            print(f"  - ... {len(stats) - 5} more")
    
    return concept_vectors

def create_random_concept_vectors(concept_vectors, device):
    """
    Create a dictionary of random concept vectors with the same structure as concept_vectors.
    Each tensor has the same shape as in concept_vectors, but with random values.
    
    Args:
        concept_vectors: Dictionary of concept vectors with structure {layer_idx: {module: {'w', 'b'}}}
        device: PyTorch device
        
    Returns:
        Dictionary with same structure as concept_vectors but with random values
    """
    random_concept_vectors = {}
    
    for layer_idx in concept_vectors:
        random_concept_vectors[layer_idx] = {}
        for module_name in concept_vectors[layer_idx]:
            # Get shapes from original concept vectors
            w_shape = concept_vectors[layer_idx][module_name]['w'].shape
            b = concept_vectors[layer_idx][module_name]['b']
            
            # Initialize with zeros first
            random_concept_vectors[layer_idx][module_name] = {
                'w': torch.zeros(w_shape, device=device),
                'b': concept_vectors[layer_idx][module_name]['b']
            }
            
            # Find non-zero positions in original weights
            non_zero_positions = torch.where(torch.norm(concept_vectors[layer_idx][module_name]['w'], dim=-1) > 0)[0]
            
            # Only add random vectors at positions that had non-zero vectors in the original
            for pos in non_zero_positions:
                random_w = torch.randn(*w_shape[1:], device=device)

                random_w_norm = torch.norm(random_w, p=2, dim=-1)  # Normalize to unit length
                random_concept_vectors[layer_idx][module_name]['w'][pos] = torch.einsum('sd,s->sd', random_w, 1 / random_w_norm) # random_w / random_w_norm
    
    return random_concept_vectors

def compute_loss_with_concept_vectors(args, val_dataset, seq_len, log_dir, device, concept_vectors=None, epsilon=0):
    """
    Compute the loss on the validation set with concept vectors.
    Similar to process_config but with concept vector projection.
    
    Returns:
        Dictionary of losses and the sequence length from the validation dataset
    """

    print(f"Detected sequence length from validation dataset: {seq_len}")
    
    # Determine vocabulary size based on task
    if args.task in ["addition", "multiplication", "binary_sorting", "parity", "majority", "majority_of_majority", "inner_product_mod2_parity"]:
        tokens_dict = CHR_DICT
    elif args.task in ["maximum_subarray", "activity_selection", "longest_common_subsequence"]:
        tokens_dict = SEQ_CHR_DICT
    elif args.task == "sorting":
        tokens_dict = SORTING_CHR_DICT
    elif args.task == "graph_maxcut":
        tokens_dict = GRAPH_MAXCUT_CHR_DICT
    elif "graph" in args.task:
        tokens_dict = GRAPH_CHR_DICT
    else:
        raise ValueError(f"Task not supported: {args.task}")
    
    # Build transformer configuration
    transformerpp_config = ModelArgs(
        max_seq_len=seq_len,
        vocab_size=len(tokens_dict.keys()) if tokens_dict else 100,
        n_layers=args.n_layer,
        n_heads=args.n_head,
        dim=args.n_embd,
        max_batch_size=args.test_batch_size,
        multiple_of=2,
        task=args.task,
    )
    
    base_model = Transformer(transformerpp_config)
    # Create a GPTTrainingModel instance
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
    if os.path.isdir(checkpoint_dir):
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if "N" not in f]
        # ckpt_files = ["N-Step-Checkpoint_epoch=0_global_step=45600.ckpt"]
        if ckpt_files:
            ckpt_filepath = os.path.join(checkpoint_dir, ckpt_files[0])
            state = torch.load(ckpt_filepath, map_location=device, weights_only=False)
            pl_model.load_state_dict(state["state_dict"])
            print(f"Loaded checkpoint from {ckpt_filepath}")
        else:
            raise FileNotFoundError("No valid checkpoint file;")
    else:
        raise FileNotFoundError("Checkpoint directory not found;")
    
    # Create DataLoader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    # Compute loss with and without concept vectors
    pl_model.eval()
    
    # 1. Baseline loss (no concept vectors)
    print("Computing baseline loss...")
    total_loss_baseline = 0
    test_correct_baseline = 0
    
    total_samples = 0
    baseline_acc_trials = []
    baseline_loss_trials = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Computing baseline loss"):
            input_ids = torch.stack(batch["ids"]).to(device).T
            attention_mask = torch.vstack(batch["attention_mask"]).to(device).T
            
            _, loss, per_token_losses = pl_model.model(
                input_ids,
                targets=input_ids[:, 1:].long(),
                loss_mask=attention_mask[:, 1:],
                capture_scores=False,
                concept_vectors={},
                epsilon=epsilon
            )
            
            batch_size = input_ids.shape[0]
            per_example_losses = per_token_losses.detach().cpu().numpy().reshape(batch_size, -1)

            per_example_losses = np.mean(per_example_losses, axis=-1)
            
            baseline_loss_trials.extend(per_example_losses.tolist())
            
            total_loss_baseline += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
            
            total_tokens = pl_model.model.params.max_seq_len // pl_model.repetitions
                
            target_token = pl_model.model.params.vocab_size - 2
            if "graph" in args.task:
                for idx in range(input_ids.shape[0]):
                    num_input_tokens = (input_ids[idx : idx + 1, :] == target_token).nonzero(as_tuple=True)[1][0]

                    remaining_tokens = total_tokens - num_input_tokens
                    input_tokens = input_ids[idx : idx + 1, :num_input_tokens]
                    label_ids = input_ids[idx : idx + 1] * attention_mask[idx : idx + 1]
                    
                    outputs = pl_model.model.generate(
                        input_tokens, remaining_tokens, temperature=1e-9, top_k=1, concept_vectors={}, epsilon=epsilon
                    ) * attention_mask[idx : idx + 1]
                    
                    
                    check = torch.all(
                        outputs == label_ids, 1
                    )
                    
                    baseline_acc_trials.extend(check.tolist())

                    test_correct_baseline += torch.sum(check).item()
            else: 
                num_input_tokens = (input_ids == target_token).nonzero(as_tuple=True)[1][0]

                remaining_tokens = total_tokens - num_input_tokens
                input_tokens = input_ids[:, :num_input_tokens]
                label_ids = input_ids * attention_mask
                
                outputs = pl_model.model.generate(
                    input_tokens, remaining_tokens, temperature=1e-9, top_k=1, concept_vectors={}, epsilon=epsilon
                ) * attention_mask
                
                
                check = torch.all(
                    outputs == label_ids, 1
                )
                baseline_acc_trials.extend(check.tolist())
                
                test_correct_baseline += torch.sum(check).item()
    
    baseline_acc = test_correct_baseline / total_samples
    baseline_loss = total_loss_baseline / total_samples
    
    print(f"Baseline Accuracy: {baseline_acc}")
    
    # 2. Loss with concept vectors (if provided)
    concept_loss = None
    concept_acc = None
    concept_trials = []
    concept_loss_trials = []
    if concept_vectors:
        print("Computing loss with concept vectors...")
        total_loss_concept = 0
        test_correct_concept = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Computing concept vector loss"):
                input_ids = torch.stack(batch["ids"]).to(device).T
                attention_mask = torch.vstack(batch["attention_mask"]).to(device).T
                
                _, loss, per_token_losses = pl_model.model(
                    input_ids,
                    targets=input_ids[:, 1:].long(),
                    loss_mask=attention_mask[:, 1:],
                    capture_scores=False,
                    concept_vectors=concept_vectors,
                    epsilon=epsilon
                )

                batch_size = input_ids.shape[0]
                per_example_losses = per_token_losses.detach().cpu().numpy().reshape(batch_size, -1)

                per_example_losses = np.mean(per_example_losses, axis=-1)
                
                concept_loss_trials.extend(per_example_losses.tolist())
                
                total_loss_concept += loss.item() * input_ids.size(0)
                
                total_tokens = pl_model.model.params.max_seq_len // pl_model.repetitions
                
                target_token = pl_model.model.params.vocab_size - 2
                
                if "graph" in args.task:
                    for idx in range(input_ids.shape[0]):
                        num_input_tokens = (input_ids[idx : idx + 1, :] == target_token).nonzero(as_tuple=True)[1][0]

                        remaining_tokens = total_tokens - num_input_tokens
                        input_tokens = input_ids[idx : idx + 1, :num_input_tokens]
                        label_ids = input_ids[idx : idx + 1] * attention_mask[idx : idx + 1]
                        
                        outputs = pl_model.model.generate(
                            input_tokens, remaining_tokens, temperature=1e-9, top_k=1, concept_vectors=concept_vectors, epsilon=epsilon
                        ) * attention_mask[idx : idx + 1]
                        
                        
                        check = torch.all(
                            outputs == label_ids, 1
                        )
                        
                        concept_trials.extend(check.tolist())

                        test_correct_concept += torch.sum(check).item()
                else: 
                    num_input_tokens = (input_ids == target_token).nonzero(as_tuple=True)[1][0]

                    remaining_tokens = total_tokens - num_input_tokens
                    input_tokens = input_ids[:, :num_input_tokens]
                    label_ids = input_ids * attention_mask
                    
                    outputs = pl_model.model.generate(
                        input_tokens, remaining_tokens, temperature=1e-9, top_k=1, concept_vectors=concept_vectors, epsilon=epsilon
                    ) * attention_mask
                    
                    check = torch.all(
                        outputs == label_ids, 1
                    )
                    concept_trials.extend(check.tolist())
                    
                    test_correct_concept += torch.sum(check).item()

        concept_loss = total_loss_concept / total_samples
        concept_acc = test_correct_concept  / total_samples
    
    # 3. Loss with random concept vectors (if provided)
    num_trials = 32
    random_loss_lst = []
    random_acc_lst = []
    random_trials = [[] for _ in range(num_trials)]
    random_loss_trials = [[] for _ in range(num_trials)]
    if concept_vectors is not None:  # We'll need this to create random vectors with same structure
        print("Computing loss with random vectors...")
        # Create random vectors with same structure
        for trial_idx in range(num_trials):
            random_vectors = create_random_concept_vectors(concept_vectors, device)
            
            total_loss_random = 0
            test_correct_random = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Computing random vector loss"):
                    input_ids = torch.stack(batch["ids"]).to(device).T
                    attention_mask = torch.vstack(batch["attention_mask"]).to(device).T
                    
                    _, loss, per_token_losses = pl_model.model(
                        input_ids,
                        targets=input_ids[:, 1:].long(),
                        loss_mask=attention_mask[:, 1:],
                        capture_scores=False,
                        concept_vectors=random_vectors,
                        epsilon=epsilon
                    )
                    
                    batch_size = input_ids.shape[0]
                    per_example_losses = per_token_losses.detach().cpu().numpy().reshape(batch_size, -1)

                    per_example_losses = np.mean(per_example_losses, axis=-1)
                    
                    random_loss_trials[trial_idx].extend(per_example_losses.tolist())
                    
                    total_loss_random += loss.item() * input_ids.size(0)
                    
                    total_loss_baseline += loss.item() * input_ids.size(0)
                    
                    total_tokens = pl_model.model.params.max_seq_len // pl_model.repetitions
                        
                    target_token = pl_model.model.params.vocab_size - 2
                    if "graph" in args.task:
                        for idx in range(input_ids.shape[0]):
                            num_input_tokens = (input_ids[idx : idx + 1, :] == target_token).nonzero(as_tuple=True)[1][0]

                            remaining_tokens = total_tokens - num_input_tokens
                            input_tokens = input_ids[idx : idx + 1, :num_input_tokens]
                            label_ids = input_ids[idx : idx + 1] * attention_mask[idx : idx + 1]
                            
                            outputs = pl_model.model.generate(
                                input_tokens, remaining_tokens, temperature=1e-9, top_k=1, concept_vectors=random_vectors, epsilon=epsilon
                            ) * attention_mask[idx : idx + 1]
                            
                            
                            check = torch.all(
                                outputs == label_ids, 1
                            )

                            test_correct_random += torch.sum(check).item()
                            random_trials[trial_idx].extend(check.tolist())
                    else: 
                            num_input_tokens = (input_ids == target_token).nonzero(as_tuple=True)[1][0]

                            remaining_tokens = total_tokens - num_input_tokens
                            input_tokens = input_ids[:, :num_input_tokens]
                            label_ids = input_ids * attention_mask
                            
                            outputs = pl_model.model.generate(
                                input_tokens, remaining_tokens, temperature=1e-9, top_k=1, concept_vectors=random_vectors, epsilon=epsilon
                            ) * attention_mask
                            
                            check = torch.all(
                                outputs == label_ids, 1
                            )
                            test_correct_random += torch.sum(check).item()
                            random_trials[trial_idx].extend(check.tolist())
            
            random_loss = total_loss_random / total_samples
            random_acc = test_correct_random / total_samples

            random_loss_lst.append(random_loss)
            random_acc_lst.append(random_acc)
            
    random_trials = np.array(random_trials)
    avg_per_question = np.mean(random_trials.T, axis=-1)
    res = stats.bootstrap((avg_per_question,),statistic=lambda x, axis: np.mean(x, axis=axis), n_resamples=10000,confidence_level=0.95,method='percentile')
    
    return {
        
        "baseline_loss": baseline_loss,
        "concept_loss": concept_loss,
        "random_loss_mean": np.mean(random_loss_lst),
        "random_loss_stderr": stats.sem(random_loss_lst),
        
        "baseline_acc": baseline_acc,
        "concept_acc": concept_acc,
        "random_acc_mean": np.mean(random_acc_lst),
        "random_acc_stderr": res.standard_error,
        "random_acc_confidence": [res.confidence_interval.low, res.confidence_interval.high],
        
        "baseline_loss_trials": baseline_loss_trials,
        "concept_loss_trials": concept_loss_trials,
        "random_loss_trials": random_loss_trials,
        
        "baseline_acc_trials": baseline_acc_trials,
        "concept_acc_trials": concept_trials,
        "random_acc_trials": random_trials.tolist(),
        
    }

def main():
    parser = argparse.ArgumentParser(description="Concept vector projection for neural networks")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing experiment results")
    parser.add_argument("--pickle_file", type=str, required=True, help="Path to the metrics pickle file")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory where checkpoints are saved")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., addition, graph_path)")
    parser.add_argument("--task_length", type=str, required=True, help="Task length")
    parser.add_argument("--method", type=str, default="normal", help="Method (normal or cot)")
    parser.add_argument("--metric", type=str, required=True, 
                       help="Metric to optimize (e.g., train_log_loss, train_acc)")
    parser.add_argument("--label_name", type=str, required=False, 
                       help="Specific labeling function to use (e.g., 'carry', 'and')")
    parser.add_argument("--target_budget_exponent", type=int, required=True, 
                       help="Target budget exponent (for 10^x FLOPs)")
    parser.add_argument("--topk", type=int, default=1,
                        help="Number of top modules to consider for each position")
    parser.add_argument("--output_json", type=str, default="concept_vector_results.json",
                       help="Output JSON file to save results")
    parser.add_argument(
        "--pool_positions",
        action="store_true",
        help="If set, positions are pooled so same vector for many positions",
    )
    args = parser.parse_args()
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Process the configuration files
    individual_runs_dir = os.path.join(
        args.directory, "1/normal/11111/4/input_reverseTrue/output_reverseTrue", args.task_length, "individual_runs"
    )
    if not os.path.isdir(individual_runs_dir):
        raise ValueError(f"Individual runs directory not found: {individual_runs_dir}")
    
    # Build a dictionary of best runs
    print("Collecting run information...")
    best_runs = collect_run_info(individual_runs_dir)
    
    # No need to filter by maximum compute since we're specifying the load compute directly
    
    # Find which (n_layer, n_embd) is best for our target_budget
    target_budget = 10**args.target_budget_exponent
    print(f"Finding best configuration for target budget: {target_budget}")
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
    
    # Find all budgets that match this best (n_layer, n_embd).
    matched_configs = find_matched_configs_across_budgets(
        best_runs, best_n_layer, best_n_embd
    )
    
    # Now load the model from the specified compute budget
    load_compute_budget = 10**args.target_budget_exponent
    if load_compute_budget in matched_configs:
        load_record = matched_configs[load_compute_budget]
        print(f"\nLoading model with compute budget={load_compute_budget}")
        print(f"  * n_layer={best_n_layer}")
        print(f"  * n_embd={best_n_embd}")
        print(f"  * val_loss={load_record['val_loss']}")
        print(f"  * file_path={load_record['file_path']}")
    else:
        print(f"No matching configuration found for compute_budget={load_compute_budget}. Exiting.")
        return
    
    # Load the metrics
    print(f"Loading metrics from {args.pickle_file}")
    metrics = load_metrics(args.pickle_file)
    
    # Find the best modules using the load_compute_exponent
    print(f"Finding top-{args.topk} modules for metric: {args.metric} using budget exponent {args.target_budget_exponent}")
    best_modules, _ = find_best_modules(metrics, args.task, args.metric, args.label_name, args.target_budget_exponent, args.topk)
    print(f"Found {len(best_modules)} best modules")
    
    # First get sequence length from validation dataset by running compute_loss_with_concept_vectors
    # with no concept vectors just to extract the sequence length
    config_path = load_record["file_path"]
    label_name = args.label_name

    print(f"\n=== Processing config: {config_path} for concept vector evaluation ===")
    old_dataset_args = create_args_from_config(config_path)

    # Load datasets
    print("Loading datasets...")
    if "test_dataset_filepath" not in metrics[str(target_budget)]:
        print("Loading base val set")
        dataset_args = old_dataset_args
        val_dataset = load_from_disk(dataset_args.val_dataset_filepath)
        
    else:
        print("Loading probe test set")
        path = metrics[str(target_budget)]['test_dataset_filepath']
        parts = path.split(os.sep)
        dataset_seed = int(parts[-3])
        
        train_filter_set_path = os.path.join(old_dataset_args.dataset_dir, 'train_numbers.npy')
        
        dataset_args = create_args_from_config(
            config_path,
            dataset_root_dir='datasets_linear_probe', 
            val_set_size=10000, 
            test_set_size=1000, 
            dataset_seed=dataset_seed,
            train_filter_set_path=train_filter_set_path
        )
        
        val_dataset = load_from_disk(metrics[str(target_budget)]['test_dataset_filepath'])

    # Get sequence length from validation dataset
    seq_len = len(val_dataset[0]["ids"])

    input_ids = np.array(val_dataset[0]["ids"])

    if args.task in ["addition", "multiplication"]:
        op_index = np.where(input_ids == 2)[0][0]
        end_index = np.where(input_ids == 4)[0][0]

        if ("carry" in label_name):
            offset = label_name.split('_')[1:]
            start_index = int(offset[0]) if offset else 1

        elif ("previous_generator" in label_name):
            start_index = 1
        else:
            start_index = 0

        start_index += 2 * op_index + 1

    elif args.task in [
        "graph_breadth_first_search",
        "graph_depth_first_search",
        "graph_path",
        "graph_topological_sort",
    ]:
        op_index = np.where(input_ids == 26)[0][0]
        end_index = np.where(input_ids == 27)[0][0]
        start_index = 0

        if label_name == "adjacency_matrix":
            start_index += op_index
        else:
            start_index += op_index + 1

    elif args.task in ["maximum_subarray", "activity_selection"]:
        dataset_args.test_batch_size = 1
        op_index = np.where(input_ids == 19)[0][0]
        end_index = op_index

        if (
            ("retrieve_start_times" in label_name)
            or ("retrieve_current_val" in label_name)
            or ("min" in label_name)
            or (label_name == "is_finish_time")
        ):
            start_index = 0
        else:
            start_index = 1

        if label_name == "min_finish_times":
            start_index += op_index
            end_index += 1

        elif label_name == "is_finish_time":
            start_index = 0
        else:
            start_index += op_index // 2
    else:
        raise ValueError("Incorrect task")  # Extend for other tasks if needed.

    # Create concept vectors with the correct sequence length
    print(f"Creating concept vectors with sequence length {seq_len}")
    concept_vectors = create_concept_vectors(
        best_modules, best_n_embd, device, seq_len, start_index, end_index, args.pool_positions
    )
    # Initialize a dictionary to store losses for each epsilon value
    all_epsilon_losses = {}

    for idx in range(1):
        epsilon = 2**idx
        # Compute losses with concept vectors
        print(
            f"Computing losses using model from load_compute_budget with epsilon={epsilon}"
        )
        losses = compute_loss_with_concept_vectors(
            dataset_args,
            val_dataset,
            seq_len,
            args.ckpt_dir,
            device,
            concept_vectors,
            epsilon,
        )

        # Store the losses for this epsilon value
        all_epsilon_losses[epsilon] = losses

        # Print results for current epsilon
        print(f"\n===== Results for epsilon={epsilon} =====")
        print(f"Baseline loss: {losses['baseline_loss']:.4f}")
        print(f"Concept vectors loss: {losses['concept_loss']:.4f}")
        print(f"Random concept vectors loss (mean): {losses['random_loss_mean']:.4f}")
        print(f"Random concept vectors loss (stderr): {losses['random_loss_stderr']:.4f}")

    # Print summary of all results
    print("\n===== Summary of Results for All Epsilon Values =====")
    for epsilon, losses in all_epsilon_losses.items():
        print(f"Epsilon={epsilon}:")
        print(f"  Baseline loss: {losses['baseline_loss']:.4f}")
        print(f"  Concept vectors loss: {losses['concept_loss']:.4f}")
        print(f"  Random concept vectors loss (mean): {losses['random_loss_mean']:.4f}")
        print(f"  Random concept vectors loss (stderr): {losses['random_loss_stderr']:.4f}")
        
        print(f"  Baseline acc: {losses['baseline_acc']:.4f}")
        print(f"  Concept vectors acc: {losses['concept_acc']:.4f}")
        print(f"  Random concept vectors acc (mean): {losses['random_acc_mean']:.4f}")
        print(f"  Random concept vectors acc (stderr): {losses['random_acc_stderr']:.4f}")

    # Save results with all epsilon values
    results = {
        "task": args.task,
        "task_length": args.task_length,
        "target_budget": target_budget,
        "load_compute_budget": load_compute_budget,
        "n_layer": best_n_layer,
        "n_embd": best_n_embd,
        "metric_used": args.metric,
        "label_name": args.label_name,
        "topk": args.topk,
        "num_concept_vectors": sum(
            len(concept_vectors[layer_idx]) for layer_idx in concept_vectors
        ),
        "epsilon_results": {
            str(epsilon): {
                "losses": losses,
            }
            for epsilon, losses in all_epsilon_losses.items()
        },
    }

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output_json}")

if __name__ == "__main__":
    main()