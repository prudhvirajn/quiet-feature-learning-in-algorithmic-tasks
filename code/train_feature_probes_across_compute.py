#!/usr/bin/env python3
"""
compute_intermediate_accuracies.py

Usage example:
    python compute_intermediate_accuracies.py \
      --directory path/to/experiment/results \
      --log_dir path/to/log/dir \
      --task_length 64 \
      --max_compute_exponent 12 \
      --target_budget_exponent 11 \
      --output_json metrics.json \
      [--do_train]

This script:
  1. Scans the experiment results (individual_runs) directory for configs.
  2. Filters runs based on a maximum compute exponent and selects the best config
     for a target budget exponent (i.e. target_budget = 10^(target_budget_exponent)).
  3. Finds all matched configs (same n_layer and n_embd) across budgets.
  4. Optionally trains them using train_from_config_filepath (if --do_train is set).
  5. For each matched config, it reconstructs the dataset (via recreate_dataset),
     builds the model (using parameters from the config), and loads the checkpoint
     (if available).
  6. Runs inference on both validation and test sets to capture intermediate activations.
  7. For each sample, finds the first occurrence of token “3” and applies a task‐specific
     labeling function (e.g. compute_carries for addition) to generate labels for each token
     position after.
  8. Trains a logistic regression classifier (separately for each layer/module and token offset)
     on validation activations and evaluates on test activations.
  9. Averages the per-token metrics (train/test accuracy and train/test log loss for logistic regression)
     and saves the results in a JSON file.
     
Make sure the following modules are available:
  - models.transformerpp (for ModelArgs, Transformer)
  - train (for GPTTrainingModel)
  - utils (for CHR_DICT, COT_CHR_DICT, SORTING_CHR_DICT, GRAPH_CHR_DICT)
  - recreate_exp (which must provide extract_dataset_info, recreate_dataset, create_args_from_config, and train_from_config_filepath)
  - datasets (for load_from_disk)
"""

import os
import json
import pickle
import argparse
from pathlib import Path
import random
from collections import deque
import time
import warnings
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_from_disk

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

from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score

# Removed KMeans and Counter imports as they are no longer used.
# from sklearn.cluster import KMeans
# from collections import Counter


# Check if y_train is one-hot encoded
def is_one_hot_encoded(y):
    # Check if all values are 0 or 1
    if not np.all(np.isin(y, [0, 1])):
        return False
    
    # Check if each row has exactly one 1 (sum equals 1)
    return np.all(np.sum(y, axis=1) == 1)


########################################
# Labeling Functions for Addition
########################################

def compute_carries(sequence, op_index):
    """
    Given a sequence of token ids (from a val_dataset sample) and the index of the operator (assumed to be token 3),
    this function computes the carry bits for an addition task.

    It assumes the sequence is structured as:
      - op1: tokens [0, op_index)
      - operator: token at index op_index (should be 3)
      - op2: tokens [op_index+1, op_index+1+len(op1)]

    Returns a list of carry bits of length len(op1)+1.
    """
    L = op_index  # op1 is the first L tokens
    op1 = sequence[:L]
    op2 = sequence[op_index + 1 : op_index + 1 + L]
    carries = [0] * (L + 1)
    for i in range(L):
        carries[i + 1] = (op1[i] & op2[i]) | ((op1[i] ^ op2[i]) & carries[i])
    return carries[1:]

def compute_ands(sequence, op_index):
    """
    Computes the bitwise AND for each position i of op1 and op2.
    Returns a list of length L (where L = op_index).
    """
    L = op_index
    op1 = sequence[:L]
    op2 = sequence[op_index + 1 : op_index + 1 + L]
    return [a & b for a, b in zip(op1, op2)]

def retrieve_op_gen(k):

    def retrieve_op(sequence, op_index):
        """
        Computes the bitwise AND for each position i of op1 and op2.
        Returns a list of length L (where L = op_index).
        """
        L = op_index
        op1 = sequence[:L]
        op2 = sequence[op_index + 1 : op_index + 1 + L]

        return op1 if k == 1 else op2

    return retrieve_op

def compute_xor(sequence, op_index):
    """
    Computes the bitwise XOR for each position i of op1 and op2.
    Returns a list of length L (where L = op_index).
    """
    L = op_index
    op1 = sequence[:L]
    op2 = sequence[op_index + 1 : op_index + 1 + L]
    return [a ^ b for a, b in zip(op1, op2)]

def compute_or(sequence, op_index):
    """
    Computes the bitwise OR for each position i of op1 and op2.
    Returns a list of length L (where L = op_index).
    """
    L = op_index
    op1 = sequence[:L]
    op2 = sequence[op_index + 1 : op_index + 1 + L]
    return [a | b for a, b in zip(op1, op2)]

def compute_indices(sequence, op_index):
    """
    Returns indices 
    """
    L = op_index
    indices = np.arange(L)

    return indices

def get_kth_partial_product_carry_function(k):
    """
    Returns a function f(sequence, op_index) -> carry_array,
    where 'carry_array' is the list of carry bits produced by
    summing the first k partial products of op1 * op2.

    - op1 = sequence[:op_index]
    - op2 = sequence[op_index+1 : op_index+1+L], with L = op_index
    - partial_product(i) = (op1 * op2[i]) << i  (bitwise shift left by i)
    
    The function f(...):
        1) Extracts op1, op2 from 'sequence'.
        2) Iteratively adds partial_product(0), partial_product(1), ...
           partial_product(k-1) into a running sum.
        3) Returns the carry array from the k-th addition step
           (or all zero carries if k=0).
    """
    def compute_kth_carries(sequence, op_index):
        # 1) Extract op1, op2
        L = op_index
        op1 = sequence[:L]
        op2 = sequence[op_index+1 : op_index+1+L]

        # We'll store the running sum in a list of bits (LSB at index 0),
        # large enough to hold up to 2*L bits if we eventually added all partial products.
        running_sum = [0] * (2 * L)
        
        # A helper function to add one partial product into running_sum
        # and capture the column-by-column carries.
        def add_with_carries(acc, addend):
            """
            In-place binary addition of 'addend' bits into 'acc' bits (both LSB at index 0).
            Returns the carry array from this addition step.

            carry_array[i+1] = carry out of column i.
            """
            length = max(len(acc), len(addend))
            carries = [0] * (length + 1)  # carry[0] = 0, carry[len] = final carry out

            carry_in = 0
            for i in range(length):
                s = carry_in
                if i < len(acc):
                    s += acc[i]
                if i < len(addend):
                    s += addend[i]
                
                acc[i] = s % 2           # new sum bit
                carry_in = s // 2       # carry out for next column
                carries[i+1] = carry_in
            
            # If there's a leftover carry beyond 'length', extend acc:
            if carry_in != 0 and length == len(acc):
                acc.append(carry_in)
                
            return carries

        # We'll keep track of the last carry array seen. If k=0, we never add anything.
        last_carries = [0] * (2*L + 1)  # a safe default with no additions

        # 2) Add partial products for i in [0..k-1], capturing the carries of the final addition
        # breakpoint()
        for i in range(k):
            bit_i = op2[i] if i < L else 0
            if bit_i == 0:
                # partial product is all zeros => adding it won't change anything
                # but let's produce a carry array of all zeros for consistency
                last_carries = [0] * (len(running_sum) + 1)
            else:
                # Build partial product i: (op1 * op2[i]) << i
                # LSB at index 0, so shifting by i means prepending i zeros.
                partial_prod_i = ([0] * i) + [op1[j] * bit_i for j in range(L)]
                
                # Add it to running_sum, capturing carries
                last_carries = add_with_carries(running_sum, partial_prod_i)

        # breakpoint()
        
        # 3) Return the carry bits from the final addition
        return last_carries[k:L + k]
    
    return compute_kth_carries

def compute_adjacency_list(sequence, op_index, is_dag=False, task="graph_breadth_first_search"):
    """
    Computes the adjacency list
    Returns a list of length L (where L = op_index).
    """
    L = op_index
    if task in ["graph_breadth_first_search", "graph_depth_first_search"]:
        start_idx = 1
    elif task in ["graph_path"]:
        start_idx = 2
    else:
        start_idx = 0

    
    edge_list = sequence[start_idx:L]
    output = sequence[L+1:]

    end_index = np.where(output == 27)[0][0]

    output = output[:end_index]

    edge_list = edge_list.reshape(-1, 2)
    graph_adjaceny_matrix = {}

    nodes = []

    for edge in edge_list:
        node1, node2 = edge

        nodes.append(node1)
        nodes.append(node2)

        if node1 not in graph_adjaceny_matrix:
            graph_adjaceny_matrix[node1] = []

        if not is_dag:
            if node2 not in graph_adjaceny_matrix:
                graph_adjaceny_matrix[node2] = []

        graph_adjaceny_matrix[node1].append(node2)
        if not is_dag: graph_adjaceny_matrix[node2].append(node1)

    unique_nodes = set(nodes)

    adjacency_matrix = np.zeros((len(unique_nodes), len(unique_nodes)))

    for node1 in graph_adjaceny_matrix:
        for node2 in graph_adjaceny_matrix[node1]:
            adjacency_matrix[node1, node2] = 1

    return np.vstack([adjacency_matrix[node] for node in output])


def compute_lexigraphical_list(sequence, op_index, is_dag=False, task="graph_breadth_first_search"):
    """
    Computes the adjacency list
    Returns a list of length L (where L = op_index).
    """
    L = op_index
    if task in ["graph_breadth_first_search", "graph_depth_first_search"]:
        start_idx = 1
    elif task in ["graph_path"]:
        start_idx = 2
    else:
        start_idx = 0

    
    edge_list = sequence[start_idx:L]
    output = sequence[L+1:]

    end_index = np.where(output == 27)[0][0]

    output = output[:end_index]

    edge_list = edge_list.reshape(-1, 2)
    graph_adjaceny_matrix = {}

    nodes = []

    for edge in edge_list:
        node1, node2 = edge

        nodes.append(node1)
        nodes.append(node2)

        if node1 not in graph_adjaceny_matrix:
            graph_adjaceny_matrix[node1] = []

        if not is_dag:
            if node2 not in graph_adjaceny_matrix:
                graph_adjaceny_matrix[node2] = []

        graph_adjaceny_matrix[node1].append(node2)
        if not is_dag: graph_adjaceny_matrix[node2].append(node1)

    unique_nodes = set(nodes)

    adjacency_matrix = np.zeros((len(unique_nodes), len(unique_nodes)))

    for node in output:
        adjacency_matrix[node, node:] = 1

    return np.vstack([adjacency_matrix[node] for node in output])


def compute_adjacency_matrix(sequence, op_index, is_dag=False, task="graph_breadth_first_search"):
    """
    Computes the bitwise OR for each position i of op1 and op2.
    Returns a list of length L (where L = op_index).
    """
    L = op_index
    if task in ["graph_breadth_first_search", "graph_depth_first_search"]:
        start_idx = 1
    elif task in ["graph_path"]:
        start_idx = 2
    else:
        start_idx = 0

    
    edge_list = sequence[start_idx:L]
    output = sequence[L+1:]

    end_index = np.where(output == 27)[0][0]

    output = output[:end_index]

    edge_list = edge_list.reshape(-1, 2)
    graph_adjaceny_matrix = {}

    nodes = []

    for edge in edge_list:
        node1, node2 = edge

        nodes.append(node1)
        nodes.append(node2)

        if node1 not in graph_adjaceny_matrix:
            graph_adjaceny_matrix[node1] = []

        if not is_dag:
            if node2 not in graph_adjaceny_matrix:
                graph_adjaceny_matrix[node2] = []

        graph_adjaceny_matrix[node1].append(node2)
        if not is_dag: graph_adjaceny_matrix[node2].append(node1)

    unique_nodes = set(nodes)

    adjacency_matrix = np.zeros((len(unique_nodes), len(unique_nodes)))

    for node1 in graph_adjaceny_matrix:
        for node2 in graph_adjaceny_matrix[node1]:
            adjacency_matrix[node1, node2] = 1

    mask = ~np.eye(adjacency_matrix.shape[0], dtype=bool)
    flattened_no_diag = adjacency_matrix[mask]

    return np.vstack([flattened_no_diag.flatten() for node in output])


def compute_queue_states(sequence, op_index, is_dag=False, task="graph_breadth_first_search"):
    """
    Computes the evolution of the traversal queue (or stack) as a binary matrix.
    Each row is a binary vector of length n (n = number of nodes) where a 1 indicates
    that the node is in the queue/stack at that step.
    
    For graph_path tasks, we assume the first element is the start (and second the target, though unused here).
    For BFS and DFS, we assume the starting node is sequence[0].
    """
    # Determine how to parse the sequence
    if task in ["graph_breadth_first_search", "graph_depth_first_search"]:
        start_idx = 1
        start_node = sequence[0]
    elif task == "graph_path":
        start_idx = 2
        start_node = sequence[0]
        # target = sequence[1]  # not used in this simulation
    else:
        start_idx = 0
        start_node = sequence[0]
    
    # Extract edge list from the sequence and reshape into pairs.
    edge_list = np.array(sequence[start_idx:op_index]).reshape(-1, 2)
    
    # Build graph as a dictionary: for each edge (node1, node2)
    graph = {}
    for node1, node2 in edge_list:
        if node1 not in graph:
            graph[node1] = []
        graph[node1].append(node2)
        if not is_dag:
            if node2 not in graph:
                graph[node2] = []
            graph[node2].append(node1)
    
    # Make sure the start node appears in the graph
    if start_node not in graph:
        graph[start_node] = []
    
    # Determine the set of all nodes (from keys and neighbors)
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)
    all_nodes = sorted(list(all_nodes))
    n = len(all_nodes)
    # Create a mapping from node value to index in our binary vector.
    node_to_index = {node: i for i, node in enumerate(all_nodes)}
    
    # Helper: given a container (queue or stack) of nodes, produce an n-length binary vector.
    def get_state_vector(container):
        vec = np.zeros(n, dtype=int)
        for node in container:
            vec[node_to_index[node]] = 1
        return vec

    states = []  # to collect each state of the queue/stack

    if task == "graph_breadth_first_search":
        # BFS: use a FIFO queue
        queue = deque([start_node])
        visited = {start_node}
        # states.append(get_state_vector(queue))  # initial state
        while queue:
            # Optionally, record state before processing the current node:
            current = queue.popleft()
            # states.append(get_state_vector(queue))
            # Enqueue unvisited neighbors.
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
            states.append(get_state_vector(queue))
    elif task == "graph_depth_first_search":
        # DFS: use a LIFO stack
        stack = [start_node]
        visited = {start_node}
        # states.append(get_state_vector(stack))
        while stack:
            current = stack.pop()
            # states.append(get_state_vector(stack))
            # Process neighbors in reverse order so that the original order is preserved in the DFS.
            for neighbor in graph.get(current, [])[::-1]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
            states.append(get_state_vector(stack))
    else:
        raise ValueError("Incorrect task")
                    
    # Combine all recorded states into a numpy array (each row is a binary vector).

    return np.vstack(states[:-1])


def compute_start_newsubarray(sequence, op_index):
    """
    Computes when a new subarray is started
    Returns a list of length L (where 1 means a new subarray is started)
    """
    L = op_index
    arr = sequence[:L]
    labels = np.zeros(L)

    max_ending_here = arr[0]
    max_so_far = arr[0]
    
    # Variables to track the indices of the best subarray
    # start = 0
    # end = 0
    temp_start = 0
    
    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
        # If extending the current subarray is worse than starting a new one
        if arr[i] > max_ending_here + arr[i]:
            max_ending_here = arr[i]
            # temp_start = i
            labels[i] = 1
        else:
            max_ending_here += arr[i]
            labels[i] = 0
        
        # Update the best subarray if we found a better sum
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            # start = temp_start
            # end = i

    # Return the subarray and the maximum sum
    return labels

def compute_intermediate_sums(sequence, op_index):
    """
    For position i, computes the best sum including i
    Returns a list of length L (where 1 means a new subarray is started)
    """
    L = op_index
    arr = sequence[:L]
    labels = np.zeros(L)

    max_ending_here = arr[0]
    max_so_far = arr[0]
    
    # Variables to track the indices of the best subarray
    # start = 0
    # end = 0
    temp_start = 0
    
    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
        # If extending the current subarray is worse than starting a new one
        if arr[i] > max_ending_here + arr[i]:
            max_ending_here = arr[i]
            # temp_start = i
        else:
            max_ending_here += arr[i]
        
        labels[i] = max_ending_here
        
        # Update the best subarray if we found a better sum
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            # start = temp_start
            # end = i

    # Return the subarray and the maximum sum
    return labels

def compute_running_maxsums(sequence, op_index):
    """
    Computes when a new subarray is started
    Returns a list of length L (where 1 means a new subarray is started)
    """
    L = op_index
    arr = sequence[:L]
    labels = np.zeros(L)

    max_ending_here = arr[0]
    max_so_far = arr[0]
    
    # Variables to track the indices of the best subarray
    # start = 0
    # end = 0
    temp_start = 0
    
    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
        # If extending the current subarray is worse than starting a new one
        if arr[i] > max_ending_here + arr[i]:
            max_ending_here = arr[i]
            # temp_start = i
        else:
            max_ending_here += arr[i]
        
        # Update the best subarray if we found a better sum
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            # start = temp_start
            # end = i
        
        labels[i] = max_so_far

    # Return the subarray and the maximum sum
    return labels

def compute_sign(sequence, op_index):
    """
    Computes when a new subarray is started
    Returns a list of length L (where 1 means a new subarray is started)
    """
    L = op_index
    arr = sequence[:L]
    labels = np.zeros(L)

    max_ending_here = arr[0]
    max_so_far = arr[0]
    
    # Variables to track the indices of the best subarray
    # start = 0
    # end = 0
    temp_start = 0
    
    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
        labels[i] = int(arr[i] >= 0)

    # Return the subarray and the maximum sum
    return labels

def compute_prev_sign(sequence, op_index):
    """
    Computes when a new subarray is started
    Returns a list of length L (where 1 means a new subarray is started)
    """
    L = op_index
    arr = sequence[:L]
    labels = np.zeros(L - 1)

    max_ending_here = arr[0]
    max_so_far = arr[0]
    
    # Variables to track the indices of the best subarray
    # start = 0
    # end = 0
    temp_start = 0
    
    # Iterate through the array starting from the second element
    for i in range(0, len(arr) - 1):
        labels[i] = int(arr[i] < 0)

    # Return the subarray and the maximum sum
    return labels

def compute_intermediate_sum_signs(sequence, op_index):
    """
    For position i, computes the best sum including i
    Returns a list of length L (where 1 means a new subarray is started)
    """
    L = op_index
    arr = sequence[:L]
    labels = np.zeros(L - 1)

    max_ending_here = arr[0]
    max_so_far = arr[0]
    
    # Variables to track the indices of the best subarray
    # start = 0
    # end = 0
    temp_start = 0
    
    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
        # If extending the current subarray is worse than starting a new one
        labels[i - 1] = int(max_ending_here < 0)
                        
        if arr[i] > max_ending_here + arr[i]:
            max_ending_here = arr[i]
            # temp_start = i
        else:
            max_ending_here += arr[i]
        
        # Update the best subarray if we found a better sum
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            # start = temp_start
            # end = i

    # Return the subarray and the maximum sum
    return labels


def compute_intermediate_maxsubarray(sequence, op_index):
    """
    Computes when a new subarray is started
    Returns a list of length L (where 1 means a new subarray is started)
    """
    L = op_index
    arr = sequence[:L]
    labels = np.zeros(L)

    max_ending_here = arr[0]
    max_so_far = arr[0]
    
    # Variables to track the indices of the best subarray
    start = 0
    end = 0
    temp_start = 0
    states = []
    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
        queue = np.zeros(19)
        # If extending the current subarray is worse than starting a new one
        if arr[i] > max_ending_here + arr[i]:
            max_ending_here = arr[i]
            temp_start = i
        else:
            max_ending_here += arr[i]
        
        # Update the best subarray if we found a better sum
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            start = temp_start
            end = i

        for idx in range(start, end+1):
            queue[arr[idx] + 9] = 1

        states.append(queue)

    # Return the subarray and the maximum sum
    return np.vstack(states)


def compute_all_intermediate_sums(sequence, op_index):
    """
    Computes when a new subarray is started
    Returns a list of length L (where 1 means a new subarray is started)
    """
    L = op_index
    arr = sequence[:L]
    labels = np.zeros(L)

    max_ending_here = arr[0]
    max_so_far = arr[0]
    
    # Variables to track the indices of the best subarray
    # start = 0
    # end = 0
    temp_start = 0
    
    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
        # If extending the current subarray is worse than starting a new one
        if arr[i] > max_ending_here + arr[i]:
            max_ending_here = arr[i]
            # temp_start = i
        else:
            max_ending_here += arr[i]
        
        labels[i] = max_ending_here
        
        # Update the best subarray if we found a better sum
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            # start = temp_start
            # end = i

    # Return the subarray and the maximum sum
    return np.vstack([labels for _ in range(len(labels))])


def compute_all_maxsums(sequence, op_index):
    """
    Computes when a new subarray is started
    Returns a list of length L (where 1 means a new subarray is started)
    """
    L = op_index
    arr = sequence[:L]
    labels = np.zeros(L)

    max_ending_here = arr[0]
    max_so_far = arr[0]
    
    # Variables to track the indices of the best subarray
    # start = 0
    # end = 0
    temp_start = 0
    
    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
        # If extending the current subarray is worse than starting a new one
        if arr[i] > max_ending_here + arr[i]:
            max_ending_here = arr[i]
            # temp_start = i
        else:
            max_ending_here += arr[i]
        
        # Update the best subarray if we found a better sum
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            # start = temp_start
            # end = i
        
        labels[i] = max_so_far

    # Return the subarray and the maximum sum
    return np.vstack([labels for _ in range(len(labels))])


def retrieve_start_times(sequence, op_index):
    L = op_index // 2  # op1 is the first L tokens
    start_times = sequence[:L]
    finish_times = sequence[L:2*L]

    return start_times


def is_finish_time(sequence, op_index):
    L = op_index // 2  # op1 is the first L tokens
    bool_finish_time = np.zeros(op_index)

    bool_finish_time[L:2*L] = 1

    return bool_finish_time

def is_prev_smaller(sequence, op_index):
    L = op_index // 2  # op1 is the first L tokens
    bool_is_prev_smaller = np.zeros(op_index)

    for idx in range(1, op_index):  
        bool_is_prev_smaller[idx] = sequence[idx - 1] <= sequence[idx]

    return bool_is_prev_smaller[1:]

def retrieve_start_times_categorical(sequence, op_index):
    L = op_index // 2  # op1 is the first L tokens
    start_times = sequence[:L]
    finish_times = sequence[L:2*L]  # Not used in this function but kept for clarity
    
    # Convert each label in start_times to a categorical vector (one-hot encoding)
    categorical_start_times = []
    for time in start_times:
        # Create a zero vector of length 10
        categorical_vector = np.zeros(10)
        # Set the position corresponding to the value to 1
        categorical_vector[time] = 1
        categorical_start_times.append(categorical_vector)
    
    return categorical_start_times

def is_greater_than_prev(sequence, op_index):
    L = op_index // 2  # op1 is the first L tokens
    start_times = sequence[:L]
    finish_times = sequence[L:2*L]  # Not used in this function but kept for clarity

    return [finish_times[idx] > finish_times[idx - 1] for idx in range(1, len(finish_times))]

def running_min(sequence, op_index):
    L = op_index // 2  # op1 is the first L tokens
    start_times = sequence[:L]
    finish_times = sequence[L:2*L]  # Not used in this function but kept for clarity

    current_min = finish_times[0]
    min_lst = []
    for idx in range(len(finish_times)):
        current_min = min(current_min, finish_times[idx])

        min_lst.append(current_min)

    return min_lst

def running_min_categorical(sequence, op_index):
    L = op_index // 2  # op1 is the first L tokens
    start_times = sequence[:L]
    finish_times = sequence[L:2*L]  # Not used in this function but kept for clarity

    current_min = finish_times[0]
    min_lst = []
    for idx in range(len(finish_times)):
        current_min = min(current_min, finish_times[idx])

        min_lst.append(current_min)

    categorical_min = []
    for min_val in min_lst:
        # Create a zero vector of length 10
        categorical_vector = np.zeros(19)
        # Set the position corresponding to the value to 1
        categorical_vector[min_val] = 1
        categorical_min.append(categorical_vector)
    
    return categorical_min

def final_min_finish_times(sequence, op_index):
    L = op_index // 2
    finish_times = sequence[L:2*L] 

    current_min = min(finish_times)

    return [current_min]

def min_finish_times_categorical_pos(sequence, op_index, cutoff=3):
    L = op_index // 2  # op1 is the first L tokens
    start_times = sequence[:L]
    finish_times = sequence[L:2*L]
    output = sequence[2*L : 2*L + 2 * cutoff + 1]

    pos_mapping = {}

    for idx in range(L):
        pos_mapping[(start_times[idx], finish_times[idx])] = idx


    finish_position_lst = []

    for idx in range(1, cutoff + 1):
        start_key, finish_key = output[2 * idx - 1], output[2 * idx]

        finish_position_lst.append(pos_mapping[(start_key, finish_key)])

    finish_retrieval = []

    for idx in range(2 * cutoff):
        feature_idx = idx // 2

        categorical_vec = np.zeros(L)

        try:

            categorical_vec[finish_position_lst[feature_idx]] = 1
        except Exception as err:
            print(err)
            breakpoint()

        finish_retrieval.append(categorical_vec)

    return np.vstack(finish_retrieval)

def min_finish_times_categorical(sequence, op_index, cutoff=3):
    L = op_index // 2  # op1 is the first L tokens
    start_times = sequence[:L]
    finish_times = sequence[L:2*L]
    output = sequence[2*L : 2*L + 2 * cutoff + 1]

    pos_mapping = {}

    for idx in range(L):
        pos_mapping[(start_times[idx], finish_times[idx])] = idx


    finish_position_lst = []

    for idx in range(1, cutoff + 1):
        start_key, finish_key = output[2 * idx - 1], output[2 * idx]

        

        finish_position_lst.append(pos_mapping[(start_key, finish_key)])

    finish_retrieval = []

    for idx in range(2 * cutoff):
        feature_idx = idx // 2

        categorical_vec = np.zeros(18)

        categorical_vec[finish_times[finish_position_lst[feature_idx]]] = 1

        finish_retrieval.append(categorical_vec)

    return np.vstack(finish_retrieval)

def min_finish_times_numerical(sequence, op_index, cutoff=3):
    L = op_index // 2  # op1 is the first L tokens
    start_times = sequence[:L]
    finish_times = sequence[L:2*L]
    output = sequence[2*L : 2*L + 2 * cutoff + 1]

    pos_mapping = {}

    for idx in range(L):
        pos_mapping[(start_times[idx], finish_times[idx])] = idx


    finish_position_lst = []

    for idx in range(1, cutoff + 1):
        start_key, finish_key = output[2 * idx - 1], output[2 * idx]

        finish_position_lst.append(pos_mapping[(start_key, finish_key)])

    finish_retrieval = []

    for idx in range(2 * cutoff):
        feature_idx = idx // 2

        finish_retrieval.append(finish_times[finish_position_lst[feature_idx]])

    return np.array(finish_retrieval)

def update_min(sequence, op_index):
    L = op_index // 2  # op1 is the first L tokens
    start_times = sequence[:L]
    finish_times = sequence[L:2*L]  # Not used in this function but kept for clarity

    current_min = finish_times[0]
    update_min = []

    for idx in range(len(finish_times)):
        if current_min > finish_times[idx]:
            current_min = finish_times[idx]
            update_min.append(1)
        else:
            update_min.append(0)

    return update_min

def retrieve_current_val(sequence, op_index):
    L = op_index  # op1 is the first L tokens
    times = sequence[:L]

    return times

def retrieve_current_val_categorical(sequence, op_index):
    L = op_index # op1 is the first L tokens
    times = sequence[:L]
    
    # Convert each label in start_times to a categorical vector (one-hot encoding)
    categorical_times = []
    for time in times:
        # Create a zero vector of length 19
        categorical_vector = np.zeros(19)
        # Set the position corresponding to the value to 1
        categorical_vector[time] = 1
        categorical_times.append(categorical_vector)
    
    return categorical_times

def compute_running_sums(sequence, op_index):
    """
    Computes the bitwise OR for each position i of op1 and op2.
    Returns a list of length L (where L = op_index).
    """
    L = op_index
    op1 = sequence[:L]
    sums = [0 for _ in range(L)]
    running_sum = 0

    group_width = L // 4

    for idx in range(L):
        if idx % group_width == 0:
            sums[idx - 1] = int(running_sum > group_width // 2)
            running_sum = 0

        running_sum = running_sum + op1[idx]

    return sums[1:]

def retrieve_four_majority(sequence, op_index):
    """
    Computes the bitwise OR for each position i of op1 and op2.
    Returns a list of length L (where L = op_index).
    """
    L = op_index
    op1 = sequence[:L]
    sums = [0 for _ in range(L)]
    running_sum = 0

    group_width = L // 4

    for idx in range(L):
        if idx % group_width == 0:
            running_sum = 0

        running_sum = running_sum + op1[idx]
        sums[idx] = running_sum

    return [[sums[idx - 1] for idx in range(group_width, L + 1, group_width)]]

# Dictionary of labeling functions per task.
LABELING_FUNCTIONS = {
    "addition": {
        "retrieve_first_op": retrieve_op_gen(1),
        "carry": compute_carries,
    },
    "multiplication": {
        "retrieve_first_op": retrieve_op_gen(1),
        "carry": get_kth_partial_product_carry_function(16),
    },
    "graph_breadth_first_search": {
        "adjacency_list": compute_adjacency_list,
        "queue": compute_queue_states,
    },
    "graph_depth_first_search": {
        "adjacency_list": compute_adjacency_list,
        "queue": compute_queue_states,
    },
    "graph_path": {
        "adjacency_list": compute_adjacency_list,
        "adjacency_matrix": compute_adjacency_matrix,
        "queue": compute_queue_states, 
    },
    "graph_topological_sort": {
        "adjacency_list": compute_adjacency_list,
        "adjacency_matrix": compute_adjacency_matrix,
    },
    "maximum_subarray": {
        "compute_prev_sign": compute_prev_sign,
        "intermediate_sums": compute_intermediate_sums,
    },
    "activity_selection": {
        "retrieve_start_times": retrieve_start_times,
    },
}

########################################
# Functions for Config Selection
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

########################################
# Inference and Activation Extraction
########################################

def run_inference(model, dataloader, device):
    """
    Run model on the dataloader to capture activations and return:
      - a list of input_id arrays (one per sample)
      - a nested dict of activations organized by layer and module.
    """
    all_input_ids = []
    # Initialize dict: activations[layer_idx][module] = list of sample arrays (each [seq_len, hidden_dim])
    activations = {}
    for layer_idx, layer in enumerate(model.model.layers):
        activations[layer_idx] = {}
        
        if hasattr(layer, "store_residual_activations"):
            activations[layer_idx]["residual"] = []

    model.eval()
    all_per_token_losses = []
    with torch.no_grad():
        for batch in dataloader:
            # Assume batch["ids"] is a list/tensor of token ids.
            input_ids = torch.stack(batch["ids"]).to(device).T  # shape: (batch_size, seq_len)
            loss_mask = torch.vstack(batch["attention_mask"]).to(device).T

            for sample in input_ids.cpu().numpy():
                all_input_ids.append(sample)
            # Forward pass (activations are captured if flag set)
            _, _, per_token_losses = model.model(
                input_ids,
                targets=input_ids[:, 1:].long(),
                loss_mask=loss_mask[:, 1:],
                capture_scores=True,
            )

            batch_size = input_ids.shape[0]

            loss_mask_counts = loss_mask[:, 1:].sum(dim=1)  # Sum along each row to count 1s
            if torch.all(loss_mask_counts == loss_mask_counts[0]):
                all_per_token_losses.append(per_token_losses.detach().cpu().numpy().reshape(batch_size, -1))

            # For each layer, extract stored activations for this batch.
            for layer_idx, layer in enumerate(model.model.layers):

                # Residual activations
                if "residual" in activations[layer_idx]:
                    act = layer.store_residual_activations[-1]
                    if hasattr(act, "cpu"):
                        res_act = act.cpu().numpy()
                    else:
                        res_act = act

                    if layer_idx == 0:
                        prev_residual_act = layer.store_residual_activations[0]

                    else:
                        prev_residual_act = activations[layer_idx - 1]["residual"]

                    for sample_act in res_act:
                        activations[layer_idx]["residual"].append(sample_act)
                    
                    layer.clear_residual_activations()

                # Attention Score
                if "attention_scores" in activations[layer_idx]:
                    layer.attention.clear_scores()

                # FFN Scores
                if "ffn_scores" in activations[layer_idx]:
                    layer.feed_forward.clear_scores()

    return all_input_ids, activations, np.vstack(all_per_token_losses) if all_per_token_losses else None

########################################
# Labeling and Activation Extraction per Sample
########################################

def extract_relative_activations_and_labels(all_input_ids, activations, task, label_name, labeling_func):
    """
    For each sample (from all_input_ids) and for each layer/module activation,
    find the first occurrence of token 2 (assumed operator). Then, call labeling_func with the
    full sequence and the operator index to obtain the labels. Finally, for each relative token
    position r (starting at 1 after the operator), extract the corresponding activation vector
    and pair it with the label.

    Returns a nested dict:
       results[layer_idx][module][relative_index] = {'X': [act_vec, ...], 'y': [label, ...]}
    """
    results = {}
    num_samples = len(all_input_ids)
    for layer_idx in activations.keys():
        results[layer_idx] = {}
        for module in activations[layer_idx].keys():
            if 'attention_scores' in module:
                sample_act =  activations[layer_idx][module][0]

                for head_idx in range(sample_act.shape[0]):
                    results[layer_idx][f"{module}:{head_idx}"] = {}
            else:
                results[layer_idx][module] = {}

    counter = 0
    for i in range(num_samples):
        input_seq = all_input_ids[i]  # a NumPy array of token ids
        # Find the first occurrence of token 2 (the operator) -- adjust if needed.
        if task in ["addition", "multiplication"]:
            op_index = np.where(input_seq == 2)[0][0]

            L = op_index  # number of tokens in op1

            if ("carry" in label_name):
                offset = label_name.split('_')[1:]
                start_index = int(offset[0]) if offset else 1

            elif ("previous_generator" in label_name):
                start_index = 1
            else:
                start_index = 0
            # Call the labeling function with the full sequence and operator index.
            labels_array = labeling_func(input_seq, op_index)
        elif task in ["graph_breadth_first_search", "graph_depth_first_search", "graph_path", "graph_topological_sort"]:
            op_index = np.where(input_seq == 26)[0][0]

            if task == "graph_topological_sort":
                is_dag = True
            else:
                is_dag = False

            L = op_index
            labels_array = labeling_func(input_seq, op_index, is_dag=is_dag, task=task)
            start_index = 0

        elif task in ["maximum_subarray", "activity_selection"]:
            op_index = np.where(input_seq == 19)[0][0]
            end_position = np.where(input_seq == 20)[0][0]


            if ("min_finish_times" in label_name) and (end_position - op_index <= 2 * 3 + 1):
                continue

            if counter >= 10000:
                break

            counter += 1
            
            if ("retrieve_start_times" in label_name) or ("retrieve_current_val" in label_name) or ("min" in label_name) or (label_name == "is_finish_time"):
                start_index = 0
            else:
                start_index = 1
            labels_array = labeling_func(input_seq, op_index)

        elif task in ["majority_of_majority"]:
            op_index = np.where(input_seq == 3)[0][0]

            if "retrieve_four_majority" in label_name:
                start_index = op_index
            else:
                start_index = 1
            labels_array = labeling_func(input_seq, op_index)
        else:
            raise ValueError("Incorrect task")  # Extend for other tasks if needed.
        
            
        # For each relative token position r (starting at 1)
        for r in range(len(labels_array)):
            abs_idx = start_index

            if task in ["addition", "multiplication"]:
                abs_idx += 2 * op_index + 1 + r
            elif task in ["graph_breadth_first_search", "graph_depth_first_search", "graph_path", "graph_topological_sort"]:
                if label_name == 'adjacency_matrix':
                    abs_idx += op_index + r
                else:
                    abs_idx += op_index + 1 + r
            elif task in ["maximum_subarray", "majority_of_majority"]:
                abs_idx += r
            elif task in ["activity_selection"]:
                if "min_finish_times" in label_name:
                    abs_idx += op_index + r
                elif (label_name in ["is_finish_time", "is_prev_smaller"]):
                    pass
                else:
                    abs_idx += op_index // 2 + r
            else:
                raise ValueError("Incorrect task")
            
            label = labels_array[r]
            for layer_idx in activations.keys():
                for module in activations[layer_idx].keys():
                    sample_act = activations[layer_idx][module][i]  # shape: (seq_len, hidden_dim)
                    if len(sample_act.shape) == 3:
                        for head_idx in range(sample_act.shape[0]):
                            if abs_idx >= sample_act[head_idx].shape[0]:
                                continue
                            act_vec = sample_act[head_idx, abs_idx]
                            if r not in results[layer_idx][f"{module}:{head_idx}"]:
                                results[layer_idx][f"{module}:{head_idx}"][r] = {"X": [], "y": []}
                            results[layer_idx][f"{module}:{head_idx}"][r]["X"].append(act_vec)
                            results[layer_idx][f"{module}:{head_idx}"][r]["y"].append(label)
                    else:
                        if abs_idx >= sample_act.shape[0]:
                            continue

                        act_vec = sample_act[abs_idx]
                        if r not in results[layer_idx][module]:
                            results[layer_idx][module][r] = {"X": [], "y": []}
                        results[layer_idx][module][r]["X"].append(act_vec)
                        results[layer_idx][module][r]["y"].append(label)
    return results

########################################
# Classifier Training and Metric Evaluation
########################################

def train_linear_models(X_train, X_test, y_train, y_test, metrics, layer_idx, module, fit_intercept=True, n_jobs=1, max_iter=1000):
    # Skip if only one class is present (or if y is empty)

    # If any y value is greater than 1, assume regression.
    if np.any(np.abs(y_train) > 1):
        # Use MultiOutputRegressor for multi-dimensional y
        if y_train.ndim > 1 and y_train.shape[1] > 1:
            # reg = MultiOutputRegressor(LinearRegression())
            raise NotImplementedError("Not supporting multi output regression")
        else:
            reg = LinearRegression()

        reg.fit(X_train, y_train)

        y_train_pred = reg.predict(X_train)
        y_test_pred = reg.predict(X_test)
        
        # Compute regression metrics: mean squared error and R²
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        metrics[layer_idx][module]["train_mse"].append(train_mse)
        metrics[layer_idx][module]["train_r2"].append(train_r2)
        metrics[layer_idx][module]["test_mse"].append(test_mse)
        metrics[layer_idx][module]["test_r2"].append(test_r2)
        
        # Store the model coefficients
        if hasattr(reg, 'coef_'):
            metrics[layer_idx][module]["coef"].append(reg.coef_.copy())
            metrics[layer_idx][module]["intercept"].append(reg.intercept_.copy())
    else:
        # Classification branch
        if y_train.ndim > 1 and y_train.shape[1] > 1:
            if is_one_hot_encoded(y_train):
                # For one-hot encoded data, use a single classifier with softmax
                clf = LogisticRegression(fit_intercept=fit_intercept, multi_class='multinomial', C = 100, max_iter=max_iter)
                
                # Convert one-hot encoded data to class labels for training
                y_train_labels = np.argmax(y_train, axis=1)
                y_test_labels = np.argmax(y_test, axis=1)
                
                clf.fit(X_train, y_train_labels)
                
                # Get predictions
                y_train_pred_labels = clf.predict(X_train)
                y_test_pred_labels = clf.predict(X_test)
                
                # Convert predictions back to one-hot format for consistent comparison
                y_train_pred = np.zeros_like(y_train)
                y_test_pred = np.zeros_like(y_test)
                for i, label in enumerate(y_train_pred_labels):
                    y_train_pred[i, label] = 1
                for i, label in enumerate(y_test_pred_labels):
                    y_test_pred[i, label] = 1

                unique_classes = np.unique(y_train_labels)
                
                # Element-wise accuracy
                train_acc = np.mean(np.all(y_train_pred == y_train, axis=1))
                test_acc = np.mean(np.all(y_test_pred == y_test, axis=1))
                
                # Log loss with predicted probabilities
                probs_train = clf.predict_proba(X_train)
                probs_test = clf.predict_proba(X_test)
                
                train_ll = log_loss(y_train_labels, probs_train, labels=unique_classes)
                test_ll = log_loss(y_test_labels, probs_test, labels=unique_classes)
                
                # Store coefficients
                metrics[layer_idx][module]["coef"].append(clf.coef_.copy())
                metrics[layer_idx][module]["intercept"].append(clf.intercept_.copy())
                
            else:
                # For multi-label data, use the original approach with valid columns
                valid_cols = [i for i in range(y_train.shape[1]) if np.unique(y_train[:, i]).size > 1]
                # Update y_train and y_test to keep only the valid columns
                y_train = y_train[:, valid_cols]
                y_test = y_test[:, valid_cols]
                
                clf = MultiOutputClassifier(LogisticRegression(fit_intercept=fit_intercept, max_iter=max_iter, C = 100))
                clf.fit(X_train, y_train)
                
                # Get predictions
                y_train_pred = clf.predict(X_train)
                y_test_pred = clf.predict(X_test)
                
                # Element-wise accuracy averaged over outputs
                train_acc = np.mean(y_train_pred == y_train)
                test_acc = np.mean(y_test_pred == y_test)
                
                # Compute log loss for each output dimension and average
                probs_train = clf.predict_proba(X_train)  # returns a list of arrays
                probs_test = clf.predict_proba(X_test)
                
                train_lls = []
                test_lls = []
                num_outputs = y_train.shape[1]
                
                for j in range(num_outputs):
                    train_lls.append(log_loss(y_train[:, j], probs_train[j]))
                    
                    if np.any(y_test[:, j] != 0):
                        test_lls.append(log_loss(y_test[:, j], probs_test[j]))
                
                train_ll = np.mean(train_lls)
                test_ll = np.mean(test_lls) if test_lls else 0.0
                
                # Store coefficients from all estimators
                coefs = np.array([estimator.coef_.copy()[0] for estimator in clf.estimators_])
                intercepts = np.array([estimator.intercept_.copy()[0] for estimator in clf.estimators_])

                metrics[layer_idx][module]["coef"].append(coefs)
                metrics[layer_idx][module]["intercept"].append(intercepts)
        else:
            clf = LogisticRegression(fit_intercept=fit_intercept, max_iter=max_iter, C = 100, n_jobs=n_jobs)
            clf.fit(X_train, y_train)
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            train_ll = log_loss(y_train, clf.predict_proba(X_train))
            test_ll = log_loss(y_test, clf.predict_proba(X_test))
            
            # Store coefficients
            metrics[layer_idx][module]["coef"].append(clf.coef_.copy())
            metrics[layer_idx][module]["intercept"].append(clf.intercept_.copy())
        
        metrics[layer_idx][module]["train_acc"].append(train_acc)
        metrics[layer_idx][module]["train_log_loss"].append(train_ll)
        metrics[layer_idx][module]["test_acc"].append(test_acc)
        metrics[layer_idx][module]["test_log_loss"].append(test_ll)

def train_and_evaluate(results_train, results_test, pool_positions=False):
    """
    For each layer/module and each relative token position r present in both training and test splits,
    train either a classification or regression model (depending on y values) and evaluate on test activations.
    
    For classification (y values not > 1):
      - Metrics: "train_acc", "train_log_loss", "test_acc", "test_log_loss"
      
    For regression (if any y value > 1):
      - Metrics: "train_mse", "train_r2", "test_mse", "test_r2"
      
    Also stores the model coefficients for both classification and regression models.
    """
    metrics = {}

    # train_activations = {}
    # test_activations = {}
    labels = {}

    if pool_positions:
        for layer_idx in results_train:
            for module in results_train[layer_idx]:
                # pool train examples from every relative position r
                pooled_train = {"X": [], "y": []}
                for r in results_train[layer_idx][module]:
                    pooled_train["X"].extend(results_train[layer_idx][module][r]["X"])
                    pooled_train["y"].extend(results_train[layer_idx][module][r]["y"])
                results_train[layer_idx][module] = {"all": pooled_train}

                # do the same for the test split
                pooled_test = {"X": [], "y": []}
                for r in results_test[layer_idx][module]:
                    pooled_test["X"].extend(results_test[layer_idx][module][r]["X"])
                    pooled_test["y"].extend(results_test[layer_idx][module][r]["y"])
                results_test[layer_idx][module] = {"all": pooled_test}

    for layer_idx in results_train.keys():
        metrics[layer_idx] = {}
        for module in results_train[layer_idx].keys():
            if module != 'residual':
                continue

            # Initialize both sets of keys. Only one branch will get filled.
            metrics[layer_idx][module] = {
                "train_acc": [],
                "train_log_loss": [],
                "test_acc": [],
                "test_log_loss": [],
                "train_mse": [],
                "train_r2": [],
                "test_mse": [],
                "test_r2": [],
                "coef": [],  # Add this key to store model coefficients
                "intercept": [],  # Add this key to store model intercepts
            }
            common_rs = set(results_train[layer_idx][module].keys()).intersection(set(results_test[layer_idx][module].keys()))
            
            for r in tqdm(common_rs, desc="Processing", unit="item"):

                # if r not in train_activations:
                #     train_activations[r] = []
                #     test_activations[r] = []
                #     labels[r] = {
                #         "train": None,
                #         "test": None
                #     }

                X_train = np.array(results_train[layer_idx][module][r]["X"])
                y_train = np.array(results_train[layer_idx][module][r]["y"])
                X_test = np.array(results_test[layer_idx][module][r]["X"])
                y_test = np.array(results_test[layer_idx][module][r]["y"])

                # train_activations[r].append(X_train)
                # test_activations[r].append(X_train)

                # labels[r]["train"] = y_train
                # labels[r]["test"] = y_test

                if y_train.size == 0 or len(np.unique(y_train)) < 2:
                    print(f"Skipping position: {r}")
                    continue
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    train_linear_models(X_train, X_test, y_train, y_test, metrics, layer_idx, module)
                    
            # Average the metrics over relative positions.
            # We'll average over all keys that have non-empty lists.
            for key in metrics[layer_idx][module]:
                if (key != "coef" and key != "intercept") and isinstance(metrics[layer_idx][module][key], list) and len(metrics[layer_idx][module][key]) > 0:
                    metrics[layer_idx][module][key] = np.array(metrics[layer_idx][module][key]) # float(np.mean(metrics[layer_idx][module][key]))
                elif (key == "coef" or key == "intercept"):
                    # Keep coefficients as a list, don't average them
                    metrics[layer_idx][module][key] = metrics[layer_idx][module][key]
                else:
                    metrics[layer_idx][module][key] = None

    return metrics

########################################
# Processing a Single Config File for Activation Analysis
########################################

def process_config(config_path, log_dir, device, random_baseline=False, train_set_size=10000, test_set_size=1000, pool_positions=False):
    """
    Given a config file and a log directory (where checkpoints are saved),
    use create_args_from_config to extract args and dataset info, build the model,
    load the checkpoint (if available), run inference on validation and test sets,
    and compute activation analysis metrics using each labeling function.

    Returns a dict with identification info and the computed metrics.
    """
    print(f"\n=== Processing config: {config_path} ===")
    old_args = create_args_from_config(config_path)

    train_filter_set_path = os.path.join(old_args.dataset_dir, 'train_numbers.npy')
    dataset_seed = random.randint(0, 65536)

    args = create_args_from_config(
        config_path,
        dataset_root_dir='datasets_linear_probe', 
        train_set_size=train_set_size, 
        test_set_size=test_set_size, 
        dataset_seed=dataset_seed,
        train_filter_set_path=train_filter_set_path
    )

    print(
        f"Extracted args: task={args.task}, task_length={args.task_length}, n_layer={args.n_layer}, n_embd={args.n_embd}, compute_budget={args.compute_budget}"
    )

    # Load datasets.
    print("Loading datasets...")
    train_dataset = load_from_disk(args.train_dataset_filepath)
    test_dataset = load_from_disk(args.test_dataset_filepath)
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

    # Load checkpoint if available.
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
    if not random_baseline:
        # Existing checkpoint loading code.
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
            if ckpt_files:
                ckpt_filepath = os.path.join(checkpoint_dir, ckpt_files[0])
                state = torch.load(ckpt_filepath, map_location=device, weights_only=False)
                pl_model.load_state_dict(state["state_dict"])
                print(f"Loaded checkpoint from {ckpt_filepath}")
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
    print("Running inference on train set for linear probe...")
    train_input_ids, train_activations, val_per_token_losses = run_inference(pl_model, train_loader, device)
    print("Running inference on test set for linear probe...")
    test_input_ids, test_activations, test_per_token_losses = run_inference(pl_model, test_loader, device)

    train_input_ids = np.array(train_input_ids)
    test_input_ids = np.array(test_input_ids)

    # Loop over each labeling function for the given task.
    all_metrics = {}
    if args.task == "multiplication":
        for idx in range(2, args.task_length + 1): # range(1, args.task_length + 1):
            LABELING_FUNCTIONS["multiplication"][f"carry_{idx}"] = get_kth_partial_product_carry_function(idx)

    if args.task in ["maximum_subarray"]:

        train_input_ids[:, :args.task_length] -= 9
        test_input_ids[:, :args.task_length] -= 9

    if args.task in LABELING_FUNCTIONS:
        for label_name, labeling_func in LABELING_FUNCTIONS[args.task].items():
            
            print(f"\n--- Processing labeling function: {label_name} ---")
            print("Extracting relative activations and labels...")
            train_activations_labels = extract_relative_activations_and_labels(
                train_input_ids, train_activations, args.task, label_name, labeling_func
            )
            test_activations_labels = extract_relative_activations_and_labels(
                test_input_ids, test_activations, args.task, label_name, labeling_func
            )
            
            print("Training classifiers and computing metrics...")
            metrics = train_and_evaluate(train_activations_labels, test_activations_labels, pool_positions=pool_positions)
            all_metrics[label_name] = metrics
    else:
        print(f"No labeling function defined for task {args.task}. Skipping.")
        return None

    info = {
        "config_path": config_path,
        "compute_budget": args.compute_budget,
        "n_layer": args.n_layer,
        "n_embd": args.n_embd,
        "metrics": all_metrics,  # Metrics now keyed by labeling function name.
        "val_per_token_losses": val_per_token_losses,
        "test_per_token_losses": test_per_token_losses,
        "dataset_seed": dataset_seed,
        "train_set_size": train_set_size,
        "test_set_size": test_set_size,
        "train_dataset_filepath": args.train_dataset_filepath,
        "test_dataset_filepath": args.test_dataset_filepath
    }
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
        help="Directory containing experiment results for a task",
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
        "--max_compute_exponent",
        type=float,
        default=float("inf"),
        help="Maximum compute exponent to consider. (Compute = 10^exponent)",
    )
    parser.add_argument(
        "--target_budget_exponent",
        type=int,
        required=False,
        default=None,
        help="Compute budget of interest. We find best (n_layer, n_embd) here.",
    ),
    parser.add_argument(
        "--ablation_only",
        action="store_true",
        help="If set, will only train / tests for target and not random baselines.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="If set, will call train_from_config_filepath for matched configs.",
    )
    parser.add_argument(
        "--output_pickle",
        type=str,
        default="metrics.pkl",
        help="Output pickle file to save metrics",
    )
    parser.add_argument(
        "--pool_positions",
        action="store_true",
        help="If set, train one probe per (layer, module, feature) pooled over all token positions.",
    )
    args = parser.parse_args()

    max_compute = 10**args.max_compute_exponent
    

    individual_runs_dir = os.path.join(
        args.directory, "1/normal/11111/4/input_reverseTrue/output_reverseTrue", args.task_length, "individual_runs"
    )
    if not os.path.isdir(individual_runs_dir):
        raise ValueError(f"Individual runs directory not found: {individual_runs_dir}")

    # 1) Build a dictionary of best runs for each (compute_budget, (n_layer, n_embd))
    best_runs = collect_run_info(individual_runs_dir)

    # Filter out runs whose budget is beyond max_compute.
    filtered_best_runs = {}
    for (budget, config_tuple), record in best_runs.items():
        if budget <= max_compute:
            filtered_best_runs[(budget, config_tuple)] = record

    # 2) Find which (n_layer, n_embd) is best for our target_budget.

    if args.target_budget_exponent:
        target_budget = 10**args.target_budget_exponent
        best_n_layer, best_n_embd, best_record = find_best_config_for_budget(
            filtered_best_runs, target_budget
        )
        if best_record is None:
            print(f"No valid runs found for compute_budget={target_budget}. Exiting.")
            return

        print(f"\nFor target budget={target_budget}, the best config is:")
        print(f"  * n_layer={best_n_layer}")
        print(f"  * n_embd={best_n_embd}")
        print(f"  * val_loss={best_record['val_loss']}")
        print(f"  * file_path={best_record['file_path']}")
    else:
        best_n_layer = None
        best_n_embd = None

    # 3) Find all budgets that match this best (n_layer, n_embd).
    matched_configs = find_matched_configs_across_budgets(
        filtered_best_runs, best_n_layer, best_n_embd
    )
    
    print("\nMatching configs for the chosen (n_layer, n_embd) across budgets:")
    for budget in sorted(matched_configs.keys()):
        record = matched_configs[budget]
        print(
            f"  Budget={budget}, val_loss={record['val_loss']}, file={record['file_path']}"
        )

    # 4) Optionally train these matched configs.
    if args.do_train:
        for budget in sorted(matched_configs.keys()):
            record = matched_configs[budget]
            config_file = record["file_path"]
            print(f"\n--- Training config for compute_budget={budget} ---")
            try:
                train_from_config_filepath(config_file, args.ckpt_dir)
            except Exception as e:
                print(f"Error while training budget={budget}: {e}")
    else:
        print("\n(do_train not set) Skipping training step. Done.")

    # 5) Process each matched config for activation analysis.
    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    overall_metrics = {}


    train_set_size = 10000

    for idx, budget in enumerate(sorted(matched_configs.keys())):
        if args.ablation_only and (target_budget != budget):
            continue

        record = matched_configs[budget]
        config_path = record["file_path"]
        print(f"\n--- Processing activation analysis for compute_budget={budget} ---")
        
        info = process_config(config_path, args.ckpt_dir, device, pool_positions=args.pool_positions, train_set_size=train_set_size)
        if info is not None:
            overall_metrics[str(budget)] = info


    print("\n--- Processing activation analysis for compute_budget=0 (random baseline) ---")
    
    # Do not train random baselines if we are storing this for ablation
    if not args.ablation_only:
        if args.target_budget_exponent:
            info = process_config(best_record["file_path"], args.ckpt_dir, device, random_baseline=True, pool_positions=args.pool_positions, train_set_size=train_set_size)
            if info is not None:
                info["compute_budget"] = 0
                overall_metrics[f"random_0"] = info
        else:
            for idx, budget in enumerate(sorted(matched_configs.keys())):
                record = matched_configs[budget]
                config_path = record["file_path"]
                # Now process the random baseline.
                info = process_config(config_path, args.ckpt_dir, device, random_baseline=True, pool_positions=args.pool_positions, train_set_size=train_set_size)
                if info is not None:
                    info["compute_budget"] = 0
                    overall_metrics[f"random_{idx}"] = info
        
    with open(args.output_pickle, "wb") as f:
        pickle.dump(overall_metrics, f)
    print(f"\nSaved overall metrics to {args.output_pickle}")

if __name__ == "__main__":
    main()