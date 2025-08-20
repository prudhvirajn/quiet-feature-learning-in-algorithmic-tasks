import argparse
import math
import os
from typing import Dict, List, Any, Callable

import numpy as np
import random
from collections import deque

from numpy.random import default_rng
from datasets import Dataset, concatenate_datasets
from tokenizers import Tokenizer

from calculate_max_samples import get_ctx, get_examples_from_compute

from prompts import (
    prompt_addition_normal,
    prompt_addition_cot,
    prompt_longest_common_subsquence,
    prompt_multiplication_normal,
    prompt_multiplication_cot,
    prompt_repeated_addition_normal,
    prompt_repeated_addition_cot,
    prompt_parity_normal,
    prompt_sorting_normal,
    prompt_binary_sorting_normal,
    prompt_pathfinding_normal,
    prompt_bfs_normal,
    prompt_dfs_normal,
    prompt_max_independent_set_normal,
    prompt_maxcut_normal,
    prompt_eulerian_circuit_normal,
    prompt_longest_path_normal,
    prompt_majority_normal,
    prompt_majority_of_majority_normal,
    prompt_inner_product_mod2_parity,
    prompt_min_spanning_tree_kruskal_normal,
    prompt_maximum_subarray_normal,
    prompt_activity_selection_normal,
    prompt_topological_sorting_normal
)
import time
import traceback
import sys

filename = "data.py"

from utils import (
    CHR_DICT,
    COT_CHR_DICT,
    SORTING_CHR_DICT,
    GRAPH_CHR_DICT,
    GRAPH_MAXCUT_CHR_DICT,
    SEQ_CHR_DICT,
    get_tokenizer,
    find_output_ranges,
    sample_bitstrings,
    sample_hexstrings,
    sample_multisets,
    sample_random_graphs,
    group_bitarrays,
    split_dataset_optimized,
)
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


def split_dataset(result, train_size, test_size, seed=None):
    random.seed(seed)
    train_set, test_set = [], []

    groups = list(result.values())
    random.shuffle(groups)

    for group in groups:
        # Decide where to place the current group based on counts
        if len(train_set) + len(group) <= train_size:
            train_set.extend(group)
        elif len(test_set) + len(group) <= test_size:
            test_set.extend(group)
        # Note: You might have a condition where you can't fit a group anymore,
        # you'll have to decide how to handle that (e.g., discard or fit into other set if possible)

    return np.array(train_set), np.array(test_set)


def write_dataset(
    task: str,
    train_set_size: int,
    val_set_size: int,
    test_set_size: int,
    task_length: int,
    output_train_filepath: str,
    output_test_filepath: str,
    seed: int,
    number_of_nodes: int = None,
    repetitions: int = 1,
    train_filter_set_path: str = None,
) -> None:
    """
    Generate and save dataset for various algorithmic tasks with optional rejection sampling.
    
    If train_filter_set_path is provided, rejection sampling is used to ensure that the
    generated samples do not overlap with the samples in the filter set.
    """
    output_dir = os.path.dirname(output_train_filepath)

    print(f"Creating directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating samples")
    start_time = time.time()

    # Load filter set if provided
    filter_set = None
    if train_filter_set_path is not None:
        print(f"Loading filter set from {train_filter_set_path}")
        filter_set = np.load(train_filter_set_path, allow_pickle=True)
        
        # Pre-process filter_set for faster lookups (convert to a set of hashable tuples)
        # Convert each sample to a hashable tuple for set operations
        filter_set = set(tuple(sample) for sample in filter_set)

    total_samples = train_set_size + val_set_size + test_set_size

    if task in ["addition", "multiplication", "parity", "majority", "majority_of_majority", "inner_product_mod2_parity"]:
        total_bits = 2 * task_length if task in ["addition", "multiplication", "inner_product_mod2_parity"] else task_length
        max_possible = 2 ** (total_bits)
        num_samples = min(max_possible, total_samples)
        
        sample = sample_with_rejection(
            lambda n, s: sample_bitstrings(repetitions * task_length, n, s),
            num_samples,
            filter_set,
            seed
        )

        start_time_split = time.time()
        train_set, test_set = split_dataset_optimized(sample, task_length, train_set_size, val_set_size + test_set_size, seed=None)
        print(f"Time to group numbers: {time.time() - start_time_split}")
        
    elif "sorting" in task:
        sample = sample_with_rejection(
            lambda n, s: sample_hexstrings(task_length, n, s),
            total_samples,
            filter_set,
            seed
        )

        test_set = sample[:test_set_size + val_set_size]
        train_set = sample[test_set_size + val_set_size:]

    elif task in ["maximum_subarray"]:
        sample = sample_with_rejection(
            lambda n, s: sample_multisets(n, task_length, -9, 9, s),
            total_samples,
            filter_set,
            seed
        )

        test_set = sample[:test_set_size + val_set_size]
        train_set = sample[test_set_size + val_set_size:]

    elif task in ["activity_selection", "longest_common_subsequence"]:
        sample = sample_with_rejection(
            lambda n, s: sample_multisets(n, 2 * task_length, 1, 9, s),
            total_samples,
            filter_set,
            seed
        )

        test_set = sample[:test_set_size + val_set_size]
        train_set = sample[test_set_size + val_set_size:]

    elif "graph" in task:
        if not number_of_nodes:
            raise ValueError("Number of nodes not provided")

        print(f"Number of nodes: {number_of_nodes}")
        number_of_edges = task_length
        
        # Determine graph type based on task
        if task in ["graph_path", "graph_longest_path", "graph_breadth_first_search", 
                   "graph_depth_first_search", "graph_min_spanning_tree_kruskal", 
                   "graph_topological_sort"]:
            graph_type = "random"
        elif task == "graph_eulerian_circuit":
            graph_type = "eulerian"
        elif task == "graph_maximum_independent_set":
            graph_type = "perfect"
        elif task == "graph_maxcut":
            graph_type = "planar"
        else:
            raise ValueError(f"Invalid graph task: {task}")
        
        sample = sample_with_rejection(
            lambda n, s: sample_random_graphs(number_of_nodes, number_of_edges, n, s, graph_type=graph_type),
            total_samples,
            filter_set,
            seed
        )

        test_set = sample[:test_set_size + val_set_size]
        train_set = sample[test_set_size + val_set_size:]

    else:
        raise ValueError(f"Invalid task: {task}")
    
    print(f"Time to sample numbers: {time.time() - start_time}")

    print("Saving samples")
    np.save(output_train_filepath, train_set)
    np.save(output_test_filepath, test_set)

def sample_with_rejection(sampling_function, num_samples, filter_set, initial_seed):
    """
    Generate samples using rejection sampling to avoid samples in filter_set.
    
    Args:
        sampling_function: A function that takes (num_samples, seed) and returns samples
        num_samples: Number of samples to generate
        filter_set: Set of samples to exclude (or None if no filtering needed)
        initial_seed: Initial random seed
        
    Returns:
        Array of samples not present in filter_set
    """
    if filter_set is None:
        # If no filter set, just use the original sampling method
        return sampling_function(num_samples, initial_seed)
    
    accepted_samples = []
    current_seed = initial_seed
    
    while len(accepted_samples) < num_samples:
        # Update seed for each batch
        current_seed += 1
        
        # Generate a batch (use a larger batch size to reduce number of calls)
        batch_size = min(2 * (num_samples - len(accepted_samples)), num_samples)
        batch = sampling_function(batch_size, current_seed)
        
        # Filter out samples that are in the filter set
        for sample in batch:
            # Convert sample to a hashable tuple format
            sample_tuple = tuple(sample)
                
            if sample_tuple not in filter_set:
                accepted_samples.append(sample)
                if len(accepted_samples) >= num_samples:
                    break
    
    return np.array(accepted_samples[:num_samples])

def get_encoding_func(
    tokenizer: Tokenizer,
    generate_text_func: Callable,
    task_length: int,
    ablations: str,
    input_reverse: bool = False,
    output_reverse: bool = False,
    skip_line: int = 1,
    number_of_nodes: int = None,
) -> Callable:
    def encode(example: dict) -> dict:
        text = generate_text_func(
            np.array(example["sample"]),
            task_length=task_length,
            ablations=ablations,
            input_reverse=input_reverse,
            output_reverse=output_reverse,
            skip_line=skip_line,
            number_of_nodes=number_of_nodes,
        )
        # example['text'] = text
        # example['type_ids'] = encodings.type_ids

        encodings = tokenizer.encode(text)

        tokens_dict = tokenizer.get_vocab()

        mask_indices = find_output_ranges(
            encodings.ids, start_token=tokens_dict["="], end_token=tokens_dict["<EOS>"]
        )

        example["sample"] = ""
        example["ids"] = encodings.ids
        example["attention_mask"] = np.zeros(len(encodings.ids), dtype=np.int8)
        example["attention_mask"][mask_indices] = 1

        return example

    return encode


def read_dataset(
    filepath: str,
    task_length: int,
    task: str,
    method: str,
    ablations: str,
    input_reverse: bool = False,
    output_reverse: bool = False,
    number_of_nodes: int = None,
    skip_line: int = 1,
) -> List:
    samples = np.load(filepath, allow_pickle=True)
    
    if "graph" not in task:
        slice_size = 2**16 // samples.shape[1]

        dataset_slices = [
            Dataset.from_dict({"sample": samples[idx : idx + slice_size]})
            for idx in range(0, samples.shape[0], slice_size)
        ]
    else:
        dataset_slices = [Dataset.from_dict({"sample": samples})]

    # Convert to Dataset
    dataset = concatenate_datasets(dataset_slices)
    method = method.lower().strip()

    if task == "addition":
        if method == "normal":
            tokenizer = get_tokenizer(CHR_DICT)
            text_gen_func = prompt_addition_normal

        elif method == "cot":
            tokenizer = get_tokenizer(COT_CHR_DICT)
            text_gen_func = prompt_addition_cot

        else:
            raise ValueError(f"Invalid method: {method}")

    elif task == "multiplication":
        if method == "normal":
            tokenizer = get_tokenizer(CHR_DICT)
            text_gen_func = prompt_multiplication_normal

        elif method == "cot":
            tokenizer = get_tokenizer(COT_CHR_DICT)
            text_gen_func = prompt_multiplication_cot

        else:
            raise ValueError(f"Invalid method: {method}")

    elif task == "parity":
        if method == "normal":
            tokenizer = get_tokenizer(CHR_DICT)
            text_gen_func = prompt_parity_normal

        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "majority":
        if method == "normal":
            tokenizer = get_tokenizer(CHR_DICT)
            text_gen_func = prompt_majority_normal

        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "majority_of_majority":
        if method == "normal":
            tokenizer = get_tokenizer(CHR_DICT)
            text_gen_func = prompt_majority_of_majority_normal

        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "inner_product_mod2_parity":
        if method == "normal":
            tokenizer = get_tokenizer(CHR_DICT)
            text_gen_func = prompt_inner_product_mod2_parity

        else:
            raise ValueError(f"Invalid method: {method}")

    elif task == "sorting":
        if method == "normal":
            tokenizer = get_tokenizer(SORTING_CHR_DICT)
            text_gen_func = prompt_sorting_normal
        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "binary_sorting":
        if method == "normal":
            tokenizer = get_tokenizer(CHR_DICT)
            text_gen_func = prompt_binary_sorting_normal
        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "maximum_subarray":
        if method == "normal":
            tokenizer = get_tokenizer(SEQ_CHR_DICT)
            text_gen_func = prompt_maximum_subarray_normal
        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "activity_selection":
        if method == "normal":
            tokenizer = get_tokenizer(SEQ_CHR_DICT)
            text_gen_func = prompt_activity_selection_normal
        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "longest_common_subsequence":
        if method == "normal":
            tokenizer = get_tokenizer(SEQ_CHR_DICT)
            text_gen_func = prompt_longest_common_subsquence
        else:
            raise ValueError(f"Invalid method: {method}")

    elif task == "graph_path":
        if not number_of_nodes:
            raise ValueError("Number of nodes not provided")

        if method == "normal":
            if number_of_nodes > 25:
                raise ValueError("Number of nodes has to be less than or equal to 25")

            tokenizer = get_tokenizer(GRAPH_CHR_DICT)
            text_gen_func = prompt_pathfinding_normal
        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "graph_breadth_first_search":
        if not number_of_nodes:
            raise ValueError("Number of nodes not provided")

        if method == "normal":
            if number_of_nodes > 25:
                raise ValueError("Number of nodes has to be less than or equal to 25")

            tokenizer = get_tokenizer(GRAPH_CHR_DICT)
            text_gen_func = prompt_bfs_normal
        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "graph_depth_first_search":
        if not number_of_nodes:
            raise ValueError("Number of nodes not provided")

        if method == "normal":
            if number_of_nodes > 25:
                raise ValueError("Number of nodes has to be less than or equal to 25")

            tokenizer = get_tokenizer(GRAPH_CHR_DICT)
            text_gen_func = prompt_dfs_normal
        else:
            raise ValueError(f"Invalid method: {method}")
    
    elif task == "graph_min_spanning_tree_kruskal":
        if not number_of_nodes:
            raise ValueError("Number of nodes not provided")

        if method == "normal":
            if number_of_nodes > 25:
                raise ValueError("Number of nodes has to be less than or equal to 25")

            tokenizer = get_tokenizer(GRAPH_CHR_DICT)
            text_gen_func = prompt_min_spanning_tree_kruskal_normal
        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "graph_topological_sort":
        if not number_of_nodes:
            raise ValueError("Number of nodes not provided")

        if method == "normal":
            if number_of_nodes > 25:
                raise ValueError("Number of nodes has to be less than or equal to 25")

            tokenizer = get_tokenizer(GRAPH_CHR_DICT)
            text_gen_func = prompt_topological_sorting_normal
        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "graph_maximum_independent_set":
        if not number_of_nodes:
            raise ValueError("Number of nodes not provided")

        if method == "normal":
            if number_of_nodes > 25:
                raise ValueError("Number of nodes has to be less than or equal to 25")

            tokenizer = get_tokenizer(GRAPH_CHR_DICT)
            text_gen_func = prompt_max_independent_set_normal
        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "graph_maxcut":
        if not number_of_nodes:
            raise ValueError("Number of nodes not provided")

        if method == "normal":
            if number_of_nodes > 25:
                raise ValueError("Number of nodes has to be less than or equal to 25")

            tokenizer = get_tokenizer(GRAPH_MAXCUT_CHR_DICT)
            text_gen_func = prompt_maxcut_normal
        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "graph_longest_path":
        if not number_of_nodes:
            raise ValueError("Number of nodes not provided")

        if method == "normal":
            if number_of_nodes > 25:
                raise ValueError("Number of nodes has to be less than or equal to 25")

            tokenizer = get_tokenizer(GRAPH_CHR_DICT)
            text_gen_func = prompt_longest_path_normal
        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "graph_eulerian_circuit":
        if not number_of_nodes:
            raise ValueError("Number of nodes not provided")

        if method == "normal":
            if number_of_nodes != 11:
                raise ValueError("Number of nodes has to be equal to 10")

            tokenizer = get_tokenizer(GRAPH_CHR_DICT)
            text_gen_func = prompt_eulerian_circuit_normal
        else:
            raise ValueError(f"Invalid method: {method}")

    elif task == "repeated_addition":
        if method == "normal":
            tokenizer = get_tokenizer(CHR_DICT)
            text_gen_func = prompt_repeated_addition_normal

        elif method == "cot":
            tokenizer = get_tokenizer(COT_CHR_DICT)
            text_gen_func = prompt_repeated_addition_cot

        else:
            raise ValueError(f"Invalid method: {method}")

    else:
        raise ValueError(f"Invalid task: {task}")

    encode = get_encoding_func(
        tokenizer,
        text_gen_func,
        task_length,
        ablations,
        input_reverse=input_reverse,
        output_reverse=output_reverse,
        skip_line=skip_line,
        number_of_nodes=number_of_nodes,
    )
    dataset = dataset.map(encode, num_proc=2)

    return dataset, tokenizer


def create_dataset(
    dataset_dir: str,
    task: str,
    method: str,
    ablations: str,
    task_length: int,
    seed: int,
    train_set_size: int,
    val_set_size: int,
    test_set_size: int,
    input_reverse: bool,
    output_reverse: bool,
    skip_line: int,
    repetitions: int = 1,
    number_of_nodes: int = None,
    train_filter_set_path: str = None,
):
    train_numpy_filepath = f"{dataset_dir}/train_numbers.npy"
    test_numpy_filepath = f"{dataset_dir}/test_numbers.npy"

    train_torch_dataset_filepath = f"{dataset_dir}/train_dataset"
    val_torch_dataset_filepath = f"{dataset_dir}/val_dataset"
    test_torch_dataset_filepath = f"{dataset_dir}/test_dataset"

    if not os.path.isdir(dataset_dir):
        os.makedirs(f"{dataset_dir}", exist_ok=True)

        start_time = time.time()
        print(f"Data Train Set Size: {train_set_size}")
        write_dataset(
            task,
            train_set_size,
            val_set_size,
            test_set_size,
            task_length,
            train_numpy_filepath,
            test_numpy_filepath,
            seed,
            repetitions=repetitions,
            number_of_nodes=number_of_nodes,
            train_filter_set_path=train_filter_set_path
        )
        print(f"Time to sample numbers: {time.time() - start_time}")

        start_time = time.time()
        # Load train and test datasets separately

        train_dataset, tokenizer = read_dataset(
            train_numpy_filepath,
            task_length,
            task,
            method,
            ablations,
            input_reverse=input_reverse,
            output_reverse=output_reverse,
            skip_line=skip_line,
            number_of_nodes=number_of_nodes,
        )

        test_dataset, _ = read_dataset(
            test_numpy_filepath,
            task_length,
            task,
            method,
            ablations,
            input_reverse=input_reverse,
            output_reverse=output_reverse,
            skip_line=skip_line,
            number_of_nodes=number_of_nodes,
        )

        print(f"Time to tokenize dataset: {time.time() - start_time}")

        test_dataset_split = test_dataset.train_test_split(
            test_size=val_set_size, seed=seed
        )
        val_dataset = test_dataset_split["test"]
        test_dataset = test_dataset_split["train"]

        print(len(train_dataset), len(val_dataset), len(test_dataset))

        train_dataset.save_to_disk(train_torch_dataset_filepath)
        val_dataset.save_to_disk(val_torch_dataset_filepath)
        test_dataset.save_to_disk(test_torch_dataset_filepath)

    else:
        print("Skipping Dataset Creation as dataset exists")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate list of numbers")
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to store datasets"
    )
    parser.add_argument(
        "--train_filter_set_path", type=str, required=False, default=None, help="Path to store datasets"
    )
    parser.add_argument(
        "--model_dim", type=int, required=True, help="Number of nodes in graph"
    )
    parser.add_argument(
        "--compute_budget", type=int, required=True, help="Number of nodes in graph"
    )
    parser.add_argument(
        "--num_layer", type=int, required=True, help="Number of nodes in graph"
    )
    parser.add_argument("--task_length", type=int, default=10, help="Size of the task")
    parser.add_argument(
        "--number_of_nodes", type=int, default=10, help="Number of nodes in graph"
    )
    parser.add_argument(
        "--train_set_size",
        type=int,
        help="Number of training examples, only used when train_filter_set_path is provided",
    )
    parser.add_argument(
        "--test_set_size", type=int, default=500000, help="Test set size"
    )
    parser.add_argument("--val_set_size", type=int, default=50000, help="Test set size")
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random generator"
    )
    parser.add_argument(
        "--task", type=str, required=False, default="addition", help="Task to perform"
    )
    parser.add_argument(
        "--method", type=str, required=True, help="Method to use for dataset generation"
    )
    parser.add_argument(
        "--iteration_ids",
        nargs="+",
        type=int,
        help="iteration id",
    )
    parser.add_argument(
        "--ablations", type=str, required=False, help="Bit String for ablations"
    )
    parser.add_argument(
        "--skip_line",
        type=int,
        required=False,
        default=1,
        help="Number of lines skipped in transcript",
    )
    parser.add_argument(
        "--input_reverse",
        action="store_true",
        help="Whether to reverse the dataset order",
    )
    parser.add_argument(
        "--output_reverse",
        action="store_true",
        help="Whether to reverse the dataset order",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of examples in a sample",
    )
    args = parser.parse_args()

    task = args.task
    method = args.method
    ablations = args.ablations
    skip_line = args.skip_line
    task_length = args.task_length
    number_of_nodes = args.number_of_nodes
    seed = args.seed
    dataset_dir = args.dataset_dir

    model_dim = args.model_dim
    compute_budget = args.compute_budget
    num_layer = args.num_layer

    train_filter_set_path = args.train_filter_set_path

    if train_filter_set_path is None:
        ctx = get_ctx(
            task, task_length, method, ablations, skip_line, number_of_nodes=number_of_nodes
        )

        train_set_size = get_examples_from_compute(
            model_dim, ctx, compute_budget, n_layers=num_layer
        )
        print(ctx, train_set_size)
    else:
        train_set_size = args.train_set_size

    print(f"Number of nodes: {number_of_nodes}")

    print(compute_budget, model_dim, num_layer)

    test_set_size = args.test_set_size
    val_set_size = args.val_set_size

    input_reverse = args.input_reverse
    output_reverse = args.output_reverse

    repetitions = args.repetitions

    print(f"Training Samples: {train_set_size}")

    create_dataset(
        dataset_dir,
        task,
        method,
        ablations,
        task_length,
        seed,
        train_set_size,
        val_set_size,
        test_set_size,
        input_reverse,
        output_reverse,
        skip_line,
        repetitions=repetitions,
        number_of_nodes=number_of_nodes,
        train_filter_set_path=args.train_filter_set_path
    )
