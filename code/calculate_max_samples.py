import math
import argparse

import numpy as np
import typer  # type: ignore

from prompts import (
    prompt_addition_normal,
    prompt_addition_cot,
    prompt_bfs_normal,
    prompt_binary_sorting_normal,
    prompt_dfs_normal,
    prompt_eulerian_circuit_normal,
    prompt_max_independent_set_normal,
    prompt_maxcut_normal,
    prompt_maximum_subarray_normal,
    prompt_multiplication_normal,
    prompt_multiplication_cot,
    prompt_parity_normal,
    prompt_sorting_normal,
    prompt_pathfinding_normal,
    prompt_longest_path_normal,
    prompt_majority_normal,
    prompt_majority_of_majority_normal,
    prompt_inner_product_mod2_parity,
    prompt_min_spanning_tree_kruskal_normal,
    prompt_activity_selection_normal,
    prompt_topological_sorting_normal
)

from utils import (
    CHR_DICT,
    COT_CHR_DICT,
    SEQ_CHR_DICT,
    SORTING_CHR_DICT,
    GRAPH_CHR_DICT,
    get_tokenizer,
    find_first_token_instance,
    sample_bitstrings,
    sample_hexstrings,
    sample_multisets,
    sample_random_graphs,
)


def get_examples_from_compute(dim, ctx, compute, n_layers=6):
    N_params = 12 * n_layers * dim * dim
    FLOPS_per_token = N_params + n_layers * ctx * dim
    Num_tokens = compute / (6 * FLOPS_per_token)
    num_examples = Num_tokens / (ctx)
    return int(math.ceil(num_examples) * 1.25)


def get_ctx(
    task: str,
    task_length: int,
    method: str,
    ablations: str = "11111",
    skip_line: int = 1,
    number_of_nodes: int = None,
):
    if task == "addition":
        samples = sample_bitstrings(task_length, 128, 42)
        if method == "normal":
            tokenizer = get_tokenizer(CHR_DICT)
            text_gen_func = prompt_addition_normal

        elif method == "cot":
            tokenizer = get_tokenizer(COT_CHR_DICT)
            text_gen_func = prompt_addition_cot
    elif task == "multiplication":
        samples = sample_bitstrings(task_length, 128, 42)
        if method == "normal":
            tokenizer = get_tokenizer(CHR_DICT)
            text_gen_func = prompt_multiplication_normal

        elif method == "cot":
            tokenizer = get_tokenizer(COT_CHR_DICT)
            text_gen_func = prompt_multiplication_cot

        else:
            raise ValueError(f"Invalid method: {method}")
    elif task == "parity":
        num_samples = min(2 ** (2 * task_length), 128)
        samples = sample_bitstrings(task_length, num_samples, 42)
        if method == "normal":
            tokenizer = get_tokenizer(CHR_DICT)
            text_gen_func = prompt_parity_normal

        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "majority":
        samples = sample_bitstrings(task_length, 128, 42)
        if method == "normal":
            tokenizer = get_tokenizer(CHR_DICT)
            text_gen_func = prompt_majority_normal

        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "majority_of_majority":
        samples = sample_bitstrings(task_length, 128, 42)
        if method == "normal":
            tokenizer = get_tokenizer(CHR_DICT)
            text_gen_func = prompt_majority_of_majority_normal

        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "inner_product_mod2_parity":
        samples = sample_bitstrings(task_length, 128, 42)
        if method == "normal":
            tokenizer = get_tokenizer(CHR_DICT)
            text_gen_func = prompt_inner_product_mod2_parity

        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "maximum_subarray":
        samples = sample_multisets(128, task_length, -9, 9, 42)
        if method == "normal":
            tokenizer = get_tokenizer(SEQ_CHR_DICT)
            text_gen_func = prompt_maximum_subarray_normal
        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "activity_selection":
        samples = sample_multisets(128, 2 * task_length, 1, 9, 42)
        if method == "normal":
            tokenizer = get_tokenizer(SEQ_CHR_DICT)
            text_gen_func = prompt_activity_selection_normal
        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif task == "longest_common_subsequence":
        samples = sample_multisets(128, 2 * task_length, 1, 9, 42)
        if method == "normal":
            tokenizer = get_tokenizer(SEQ_CHR_DICT)
            text_gen_func = prompt_activity_selection_normal
        else:
            raise ValueError(f"Invalid method: {method}")
        
    elif "sorting" in task:
        samples = sample_hexstrings(task_length, 128, 42)
        
        if task == "sorting":
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
        else:
            raise ValueError(f"Invalid task: {task}")
        
    elif "graph" in task:
        if not number_of_nodes:
            raise ValueError("Number of nodes not provided")

        number_of_edges = task_length
        num_samples = 128
        
        if task in ["graph_path", "graph_longest_path", "graph_breadth_first_search", "graph_depth_first_search", "graph_min_spanning_tree_kruskal", "graph_topological_sort"]:
            samples = sample_random_graphs(
                number_of_nodes, number_of_edges, num_samples, seed = 42, graph_type = "random"
            )
            
            if task == "graph_path":
                if not number_of_nodes:
                    raise ValueError("Number of nodes not provided")

                if method == "normal":
                    if number_of_nodes > 25:
                        raise ValueError("Number of nodes has to be less than or equal to 25")

                    tokenizer = get_tokenizer(GRAPH_CHR_DICT)
                    text_gen_func = prompt_pathfinding_normal
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
                
        elif task == "graph_eulerian_circuit":
            samples = sample_random_graphs(
                number_of_nodes, number_of_edges, num_samples, seed = 42, graph_type = "eulerian"
            )
            
            if not number_of_nodes:
                raise ValueError("Number of nodes not provided")

            if method == "normal":
                if number_of_nodes != 11:
                    raise ValueError("Number of nodes has to be equal to 10")

                tokenizer = get_tokenizer(GRAPH_CHR_DICT)
                text_gen_func = prompt_eulerian_circuit_normal
            else:
                raise ValueError(f"Invalid method: {method}")
        
        elif task == "graph_maximum_independent_set":
            samples = sample_random_graphs(
                number_of_nodes, number_of_edges, num_samples, seed = 42, graph_type = "perfect"
            )
            
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
            samples = sample_random_graphs(
                number_of_nodes, number_of_edges, num_samples, seed = 42, graph_type = "planar"
            )
            
            if not number_of_nodes:
                raise ValueError("Number of nodes not provided")

            if method == "normal":
                if number_of_nodes > 25:
                    raise ValueError("Number of nodes has to be less than or equal to 25")

                tokenizer = get_tokenizer(GRAPH_CHR_DICT)
                text_gen_func = prompt_maxcut_normal
            else:
                raise ValueError(f"Invalid method: {method}")
        
    else:
        raise ValueError(f"Task is incorrect: {task}")

    average_ctx = 0
    for sample_array in samples:
        example = text_gen_func(
            np.array(sample_array),
            task_length=task_length,
            ablations=ablations,
            skip_line=skip_line,
            number_of_nodes=number_of_nodes,
        )
        
        encodings = tokenizer.encode(example)
        
        tokens_dict = tokenizer.get_vocab()
        average_ctx += (
            find_first_token_instance(encodings.ids, end_token=tokens_dict["<EOS>"]) + 1
        )

    return average_ctx / len(samples)


def main(
    bit_length: int = 10,
    task: str = "addition",
    method: str = "normal",
    skip_line: int = 1,
    model_dim: int = 24,
    compute: int = int(1e12),
    ablations: str = "11111",
    n_layers: int = 6,
):
    ctx = get_ctx(task, bit_length, method, ablations, skip_line)
    max_samples = get_examples_from_compute(model_dim, ctx, compute, n_layers=n_layers)

    print(ctx, max_samples)


if __name__ == "__main__":
    # typer.run(main)

    print(main(32, task="sorting", model_dim=128, compute=int(1e12), n_layers=16))
