# ===== Standard Library =====
import csv
import json
import math
import os
import random
import re
import secrets
import time
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from math import comb
from typing import Any, Dict

# ===== Third-Party Libraries =====
import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LinearRegression, RANSACRegressor  # type: ignore
import torch
import torch.optim.lr_scheduler as lr_scheduler
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tqdm import tqdm

NUM_SORTING_ALPHABET = 16 # 512
CHR_DICT = {"0": 0, "1": 1, "+": 2, "=": 3, "<EOS>": 4}
COT_CHR_DICT = {"0": 0, "1": 1, "+": 2, "=": 3, "<EOS>": 4}

SEQ_CHR_DICT = {chr(i + 65): i for i in range(19)} # {str(i): i for i in range(NUM_SORTING_ALPHABET) }
SEQ_CHR_DICT["="] = 19
SEQ_CHR_DICT["<EOS>"] = 20

SORTING_CHR_DICT = {str(i): i for i in range(10)} # {str(i): i for i in range(NUM_SORTING_ALPHABET) }
SORTING_CHR_DICT[","] = 10
SORTING_CHR_DICT["="] = 11
SORTING_CHR_DICT["<EOS>"] = 12

GRAPH_CHR_DICT = {chr(i + 65): i for i in range(26)}
GRAPH_CHR_DICT["="] = 26
GRAPH_CHR_DICT["<EOS>"] = 27

GRAPH_MAXCUT_CHR_DICT = {chr(i + 65): i for i in range(26)}
GRAPH_MAXCUT_CHR_DICT["|"] = 26
GRAPH_MAXCUT_CHR_DICT["="] = 27
GRAPH_MAXCUT_CHR_DICT["<EOS>"] = 28

GRAPHS_FILEDIR = "graphs"


def get_tokenizer(dictionary: Dict) -> Any:
    tokenizer = Tokenizer(BPE())
    tokenizer.add_tokens([k for k, v in dictionary.items()])

    return tokenizer


def find_first_token_instance(token_indices, end_token=4):
    token_indices = torch.Tensor(token_indices)
    # Find all indices of end token
    indices_of_end_token = (token_indices == end_token).nonzero(as_tuple=True)[0]
    indices_of_end_token = np.array(indices_of_end_token)

    indices_of_end_token.sort()

    return indices_of_end_token[0]


def find_output_ranges(token_indices, start_token=3, end_token=4):
    token_indices = torch.Tensor(token_indices)
    # Find all indices of end token
    indices_of_end_token = (token_indices == end_token).nonzero(as_tuple=True)[0]

    # Initialize the end of the previous end token to -1
    prev_end = -1

    # Find the first occurrence of start token before each end token and after the previous end token
    ranges = []
    for index in indices_of_end_token:
        # Search for 3s in the range after the previous 4 and before the current 4
        indices_of_start_token = (
            token_indices[prev_end + 1 : index] == start_token
        ).nonzero(as_tuple=True)[0]

        if indices_of_start_token.numel() > 0:
            first_start_token_before_current_end_token = (
                indices_of_start_token[0].item() + prev_end + 1
            )
            ranges.append((first_start_token_before_current_end_token, index.item()))
        prev_end = index.item()

    # Get the column indices within the ranges
    column_indices = [list(range(start, end + 1)) for start, end in ranges]

    return np.concatenate(column_indices)


def bitarr2string(x):
    return "".join(["1" if i else "0" for i in x])


def digitarr2string(x):
    return "".join([str(i) for i in x])


def get_compute(run_id):
    run_dir = [i for i in os.listdir("./wandb") if run_id in i]
    if not run_dir:
        return None, None
    wandb_dir = f"./wandb/{run_dir[0]}/files/"
    summary_file = os.path.join(wandb_dir, "wandb-summary.json")

    with open(summary_file, "r") as file:
        summary = json.load(file)
        total_compute = summary["total_compute"]

    return total_compute


def get_train_loss_and_steps(run_id):
    """
    Retrieve the train_loss from wandb using the entity, project name, and run_id.
    """
    run_dir = [i for i in os.listdir("./wandb") if run_id in i]
    if not run_dir:
        return None, None
    wandb_dir = f"./wandb/{run_dir[0]}/files/"
    summary_file = os.path.join(wandb_dir, "wandb-summary.json")

    with open(summary_file, "r") as file:
        summary = json.load(file)
        train_loss = summary["train_loss_epoch"]
        early_stopped_step = summary["trainer/global_step"]

    return train_loss, early_stopped_step


def get_steps_from_compute(dim, ctx, compute, batch_size, n_layers=6):
    N_params = 12 * n_layers * dim * dim
    FLOPS_per_token = N_params + n_layers * ctx * dim
    Num_tokens = compute / (6 * FLOPS_per_token)
    total_steps = Num_tokens / (batch_size * ctx)
    return math.ceil(total_steps)


def get_compute_from_steps(dim, ctx, bit_length, total_steps, batch_size, n_layers=6):
    N_params = 12 * n_layers * dim * dim
    FLOPS_per_token = N_params + n_layers * ctx * dim
    Num_tokens = total_steps * (batch_size * ctx)
    compute = Num_tokens * (6 * FLOPS_per_token)
    return compute


def get_bs_from_compute(dim, ctx, bit_length, compute, total_steps, n_layers=6):
    N_params = 12 * n_layers * dim * dim
    FLOPS_per_token = N_params + n_layers * ctx * dim
    Num_tokens = compute / (6 * FLOPS_per_token)
    batch_size = Num_tokens / (total_steps * ctx)
    return math.ceil(batch_size)


def get_max_loss(bit_length, repetitions, ctx):
    masked_tokens = (2 * bit_length + 1) * repetitions
    return -math.log(0.5) / (ctx - masked_tokens)


def compute_ewma(data, beta=0.9):
    ewma_list = []
    ewma = 0  # Starting point for the EWMA

    for value in data:
        ewma = beta * ewma + (1 - beta) * value
        ewma_list.append(ewma)

    return ewma_list


def LOG_to_CSV(file_in, file_out):
    # Open your log file
    with open(file_in, "r") as log_file:
        # Prepare a dictionary to hold your parsed values
        parsed_dict = {}

        # Read the log file line by line
        for line in log_file.readlines():
            # Use regular expression to find the desired values in each line
            match = re.search(r"(\d+) step (GB_small_[01]|GB_big): ([\d.]+)", line)
            if match:
                step, label, value = match.groups()
                step = int(step)

                # Initialize dictionary for the step if not already initialized
                if step not in parsed_dict:
                    parsed_dict[step] = {}

                if label not in parsed_dict[step]:
                    parsed_dict[step][label] = []
                # Store the value in the dictionary
                parsed_dict[step][label].append(float(value))

        # print(parsed_dict)
        # Now, let's write the parsed values to a CSV file
        headers = ["step", "GB_small_0", "GB_small_1", "GB_big"]
        with open(file_out, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)

            # Write the header row
            writer.writeheader()

            # Write the data rows
            for step, values in parsed_dict.items():
                n = len(values["GB_small_0"])
                for i in range(n):
                    row = {"step": step}
                    for h in headers[1:]:
                        row[h] = values[h][i]
                    writer.writerow(row)


def determine_critical_batch_size(log_filepath: str, B_big: int, num_gpus: int):
    csv_filepath = f'{".".join(log_filepath.split(".")[:-1])}.csv'
    LOG_to_CSV(log_filepath, csv_filepath)

    big_list, small_0_list, small_1_list = [], [], []

    with open(csv_filepath, "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            small_0_list.append(row[1])
            small_1_list.append(row[2])
            big_list.append(row[3])

    big_list, small_0_list, small_1_list = (
        big_list[1:],
        small_0_list[1:],
        small_1_list[1:],
    )

    Gradient_norm_estimator = []

    for i in range(len(big_list)):
        e0 = (1 / (num_gpus - 1)) * (
            num_gpus * float(big_list[i]) - float(small_0_list[i])
        )
        e1 = (1 / (num_gpus - 1)) * (
            num_gpus * float(big_list[i]) - float(small_1_list[i])
        )

        Gradient_norm_estimator.append((e0 + e1) / 2)

    cov_trace_estimator = []

    for i in range(len(big_list)):
        e0 = (B_big / (num_gpus - 1)) * (-float(big_list[i]) + float(small_0_list[i]))
        e1 = (B_big / (num_gpus - 1)) * (-float(big_list[i]) + float(small_1_list[i]))

        cov_trace_estimator.append((e0 + e1) / 2)

    Gradient_norm_estimator_smooth = compute_ewma(Gradient_norm_estimator, beta=0.99)
    cov_trace_estimator_smooth = compute_ewma(cov_trace_estimator, beta=0.99)
    B_noise = [
        S / G
        for S, G in zip(cov_trace_estimator_smooth, Gradient_norm_estimator_smooth)
    ]

    B_noise_filtered = [i for i in B_noise if i > 0]

    X = np.arange(len(B_noise_filtered)).reshape(-1, 1)
    y = np.array(B_noise_filtered)

    ransac = RANSACRegressor(LinearRegression(), max_trials=1000)
    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_

    line_y_ransac = ransac.predict(X)

    return np.mean(line_y_ransac), np.mean(inlier_mask)


class WarmupCosineAnnealingLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_step=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min

        # Create internal schedulers
        self.warmup_scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps
        )
        self.cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min
        )

        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_step)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return self.warmup_scheduler.get_lr()
        else:
            # Ensure that the cosine annealing starts after warmup
            self.cosine_scheduler.last_epoch = self.last_epoch - self.warmup_steps
            return self.cosine_scheduler.get_lr()

    def step(self, step=None):
        if step is None:
            step = self.last_epoch + 1
        self.last_epoch = step
        if step < self.warmup_steps:
            self.warmup_scheduler.step(step)
        else:
            self.cosine_scheduler.step(step - self.warmup_steps)


class ConstantLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, learning_rate, last_epoch=-1):
        self.learning_rate = learning_rate

        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.learning_rate

    def step(self, epoch=None):
        self.last_epoch = self.last_epoch + 1


class WarmupConstantLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.lr = lr

        # Create internal schedulers
        self.warmup_scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        self.constant_scheduler = ConstantLR(optimizer, lr)

        super(WarmupConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return self.warmup_scheduler.get_lr()
        else:
            return self.constant_scheduler.get_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            self.warmup_scheduler.step(epoch)
        else:
            self.constant_scheduler.step(epoch)
        self.last_epoch = self.last_epoch + 1


def linear_warmup_scheduler(current_step, max_lr, warmup_steps, optimizer):
    lr = (current_step / warmup_steps) * (max_lr)

    for param in optimizer.param_groups:
        param["lr"] = lr

    return optimizer


def half_adder(a, b):
    s = a ^ b
    c = a & b

    return s, c


def binary_adder(arr1, arr2):
    # Make sure both input arrays have the same shape
    if len(arr1) != len(arr2):
        raise ValueError("Input arrays must have the same shape")

    # Initialize variables to store the sum and carry
    sum_array = np.zeros(len(arr1) + 1, dtype=np.uint8)
    carry = 0

    # Iterate through the arrays and perform binary addition
    for i in range(len(arr1) - 1, -1, -1):
        # Calculate the sum bit
        sum_bit = arr1[i] ^ arr2[i] ^ carry

        # Calculate the carry bit
        carry = (arr1[i] & arr2[i]) | (carry & (arr1[i] ^ arr2[i]))

        # Store the sum bit in the result array
        sum_array[i + 1] = sum_bit

    # If there's a final carry after the loop, append it as the most significant bit
    if carry:
        sum_array[0] = carry

    return sum_array

def longest_common_subsequence(X, Y):
    """
    Compute the longest common subsequence (LCS) of strings X and Y.
    
    Returns:
        A array representing one LCS of X and Y.
    """
    m, n = len(X), len(Y)
    # dp[i][j] holds the length of LCS of X[:i] and Y[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Build the dp table
    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    # Reconstruct one LCS from the dp table
    i, j = m, n
    lcs_chars = []
    while i > 0 and j > 0:
        if X[i-1] == Y[j-1]:
            lcs_chars.append(X[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    return lcs_chars[::-1]

def solve_activity_selection(start_times, end_times):
    """
    Solve the activity selection problem using a greedy approach.
    :param start_times: numpy.ndarray of start times for the activities.
    :param end_times: numpy.ndarray of end times for the activities.
    :return: List of indices of selected activities.
    
    # Example usage:
    start_times = np.array([1, 2, 4, 6, 5])
    end_times = np.array([3, 5, 6, 8, 7])
    selected_indices = solve_activity_selection(start_times, end_times)
    print("Selected activity indices:", selected_indices)
    """

    # Ensure inputs are 1D numpy arrays
    start_times = np.asarray(start_times).flatten()
    end_times = np.asarray(end_times).flatten()

    # Pair the start and end times along with their indices and sort by end time
    activities = sorted(enumerate(zip(start_times, end_times)), key=lambda x: x[1][1])

    # Initialize the list of selected activity indices
    selected_activities = []
    last_end_time = 0

    # Iterate through sorted activities
    for index, (start, end) in activities:
        if start >= last_end_time:
            # Select the activity
            selected_activities.append(start)
            selected_activities.append(end)
            last_end_time = end

    return selected_activities


def binary_multiplication(arr1, arr2):
    # Make sure both input arrays have the same shape
    if len(arr1) != len(arr2):
        raise ValueError("Input arrays must have the same shape")

    a = int(bitarr2string(arr1), 2)
    b = int(bitarr2string(arr2), 2)

    result = a * b

    result_bits = np.array(list(bin(result)[2:].zfill(2 * len(arr1))), dtype=arr1.dtype)

    return result_bits


def remove_model_prefix(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("model.", "", 1)  # Only replace the first occurrence
        new_state_dict[new_key] = value
    return new_state_dict


def sample_bitstrings(task_length: int, num_samples: int, seed: int = 42) -> np.array:
    """
    return: a 2d bit np array with size (num_samples, 2*task_length)
    """

    max64bit_num = 2**63 - 1 if task_length >= 32 else 2 ** (2 * task_length)

    if task_length >= 32:
        num_batch = math.ceil((task_length * 2 / 64))
        number_stack = []
        num_samples = num_samples + 1 if num_samples % 2 else num_samples

        for _ in range(num_batch):
            rng = default_rng(seed)
            cur_numbers = rng.choice(max64bit_num, size=num_samples // 2, replace=False)
            new_numbers = (
                np.hstack([cur_numbers, -cur_numbers])
                .reshape((num_samples, 1))
                .view(np.uint8)
            )
            numbers = np.unpackbits(new_numbers, axis=1)

            number_stack.append(numbers)
            seed += 1

        numbers = np.hstack(number_stack)

    else:
        rng = default_rng(seed)
        cur_numbers = (
            rng.choice(max64bit_num, size=num_samples, replace=False)
            .reshape((num_samples, 1))
            .view(np.uint8)
        )
        numbers = np.unpackbits(cur_numbers, axis=1, bitorder="little")

    numbers = numbers[:, : 2 * task_length]
    return numbers


def sample_digitstrings(task_length: int, num_samples: int, seed: int = 42) -> np.array:
    if task_length > 18:
        raise ValueError("We don't sampling digit strings for task length > 18")

    rng = default_rng(seed)

    samples = rng.choice(10**task_length, size=5 * num_samples, replace=False).reshape(
        (num_samples, 1)
    )

    return samples

@lru_cache(None)
def nCr(n, r):
    return math.comb(n, r)

def decode_combination(N, L, idx, binom_table, min_val):
    """
    Decode a 0-based index 'idx' into the corresponding sorted multiset 
    of length L drawn from [0, N-1], in lexicographic order.
    """
    # We’ll build the combination in a list, from the smallest element to largest (with repeats).
    result = []
    
    # 'current_min' is the smallest value we are still allowed to use 
    # (i.e., we won't go backwards).
    current_min = 0
    remaining_length = L

    while remaining_length > 0:
        # How many combinations start with 'current_min'?
        # If the first element is 'current_min', then the rest is a multiset 
        # of length (remaining_length - 1) from [current_min..N-1].
        # count_if_chosen = nCr((N - current_min) + (remaining_length - 1) - 1, remaining_length - 1)
        count_if_chosen = binom_table[(N - current_min) + (remaining_length - 1) - 1][remaining_length - 1]
        
        if idx < count_if_chosen:
            # That means the first element is current_min.
            result.append(current_min + min_val)
            remaining_length -= 1
            # We stay at 'current_min' for the next element because repeats are allowed,
            # but effectively the next "minimum" is still current_min (not current_min+1).
        else:
            # Skip all combinations that start with 'current_min' and move on.
            idx -= count_if_chosen
            current_min += 1

    return result

def big_sample_small_k(total, k, seed):
    """
    Returns k distinct random integers from [0..total-1].
    Works well if k is much smaller than total.
    """
    random.seed(seed)
    chosen = set()
    while len(chosen) < k:
        # secrets.randbelow(total) gives a random integer in [0..total-1],
        # even if total is a Python 'int' bigger than 2**63.
        r = random.randint(0, total - 1)
        chosen.add(r)
    return list(chosen)

def sample_multisets(k, L, min_val, max_val, seed):
    """
    Uniformly sample k distinct combinations (multisets) of length L from [min_val, max_val],
    returning each as a sorted list. No two sampled multisets are the same,
    and permutations of the same multiset are considered identical.
    """
    N = max_val - min_val + 1
    # Total number of multisets
    total = nCr(N + L - 1, L)

    random.seed(seed)
    
    if k > total:
        raise ValueError(f"Requested {k} combos but only {total} possible.")
    
    # Pick k distinct indices from [0 .. total-1]
    # (This uses an algorithm under the hood that doesn’t require building the entire range in memory.)
    start_time = time.time()
    chosen_indices = big_sample_small_k(total, k, seed) # random.sample(range(total), k)
    print(f"Time to sample indices: {time.time() - start_time}")

    # binom_table[n][r] = C(n, r)
    binom_table = [[0]*(N+L+1) for _ in range(N+L+1)]
    binom_table[0][0] = 1
    for i in range(1, N+L+1):
        binom_table[i][0] = 1
        for j in range(1, i+1):
            binom_table[i][j] = binom_table[i-1][j-1] + binom_table[i-1][j]
    
    start_time = time.time()
    combos_array = np.empty((len(chosen_indices), L), dtype=np.int64)

    # Decode each index into the corresponding multiset
    for i, rank in enumerate(chosen_indices):
        # Decode into [0..N-1], then shift each element by min_val
        decoded = decode_combination(N, L, rank, binom_table, min_val)
        random.shuffle(decoded)
        combos_array[i, :] = decoded
    
    print(f"Time to decode indices: {time.time() - start_time}")
    
    return combos_array

def sample_hexstrings(task_length: int, num_samples: int, seed: int = 42) -> np.array:
    # Define the hex alphabet
    alphabet = [str(i) for i in range(NUM_SORTING_ALPHABET)]
    
    random.seed(seed)
    # Sample unique multisets and time the function
    multisets = sample_unique_multisets(num_samples, task_length, len(alphabet), seed)

    dataset = deque()
    # Display the results
    for idx, counts in enumerate(multisets):
        multiset = []
        for count, letter in zip(counts, alphabet):
            multiset.extend([letter] * count)
        random.shuffle(multiset)
        hexstring = ','.join(multiset)  # Sorted to represent the multiset
        dataset.append(hexstring)

    return np.array(dataset).reshape((len(dataset), 1))

def find_max_subarray(arr):
    """
    Returns the contiguous subarray of 'arr' which has the maximum sum,
    along with the sum of that subarray.
    """
    # Initialize with the first element
    max_ending_here = arr[0]
    max_so_far = arr[0]
    
    # Variables to track the indices of the best subarray
    start = 0
    end = 0
    temp_start = 0
    
    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
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
    
    # Return the subarray and the maximum sum
    return arr[start:end+1].copy(), max_so_far

# def sample_hexstrings(task_length: int, num_samples: int, seed: int = 42) -> np.array:
#     # if task_length > 14:
#     #     raise ValueError("We don't sampling digit strings for task length > 14")

#     rng = default_rng(seed)
#     unique_samples = deque(maxlen=num_samples)

#     for _ in range(3):
#         samples = rng.choice(int('0x'+'F'*task_length, 16), size=10 * num_samples, replace=False)
#         hexes = np.vectorize(hex)(samples) 
#         hash_map = defaultdict(bool)

#         for sample in hexes:
#             hash_key = ''.join(sorted(str(sample)))
            
#             if not hash_map[hash_key]:
#                 hash_map[hash_key] = True
#                 unique_samples.append(sample)
            
#             if len(unique_samples) == num_samples:
#                 break
        
#         if len(unique_samples) == num_samples:
#             break
        
#         unique_samples.clear()
        
#     if len(unique_samples) < num_samples:
#         num_samples = unique_samples
#         print("Could not generate enough unique samples")

#     return np.array(unique_samples).reshape((num_samples, 1))


def sample_random_graphs(
    number_of_nodes: int, number_of_edges: int, num_samples: int, seed: int = 42, graph_type: str = 'random'
) -> np.array:
    import sage.all  # type: ignore
    from sage.graphs.graph import Graph  # type: ignore
    
    if graph_type == 'random':
        graph_filepath = os.path.join(GRAPHS_FILEDIR, f"graph{number_of_nodes}c.g6")
    elif graph_type == 'eulerian':
        graph_filepath = os.path.join(GRAPHS_FILEDIR, f"eul{number_of_nodes}.c.g6")
    elif graph_type == 'perfect':
        graph_filepath = os.path.join(GRAPHS_FILEDIR, f"perfect{number_of_nodes}.g6")
    elif graph_type == 'planar':
        graph_filepath = os.path.join(GRAPHS_FILEDIR, f"planar_conn.{number_of_nodes}.g6")
        
    if not os.path.exists(graph_filepath):
        raise ValueError(f"Graph file does not exist: {graph_filepath}")
    
    with open(graph_filepath, "r") as file:
        graphs = file.readlines()
        
    num_samples = min(len(graphs), num_samples)

    rng = default_rng(seed)

    choices = rng.choice(len(graphs), size=num_samples, replace=False)
    samples = deque()

    for choice in choices:
        graph = Graph(graphs[choice].strip())

        perm = list(range(number_of_nodes))
        rng.shuffle(perm)

        mapping = { old: perm[old] for old in range(number_of_nodes) }
        graph.relabel(mapping, inplace=True)
        
        edge_list = [
            node for edge in graph.edges(sort=True, labels=None) for node in edge
        ]
        
        samples.append(edge_list)
        
    return np.array(samples)

def sample_random_euler_graphs(
    number_of_nodes: int, number_of_edges: int, num_samples: int, seed: int = 42
) -> np.array:
    import sage.all  # type: ignore
    from sage.graphs.graph import Graph  # type: ignore
    
    with open(os.path.join(GRAPHS_FILEDIR, f"eul{number_of_nodes}c.g6"), "r") as file:
        graphs = file.readlines()
        
    if len(graphs) < num_samples:
        raise ValueError("Could not generate enough unique samples")

    rng = default_rng(seed)

    choices = rng.choice(len(graphs), size=num_samples, replace=False)
    samples = deque()

    for choice in choices:
        graph = Graph(graphs[choice].strip())
        
        edge_list = [
            node for edge in graph.edges(sort=True, labels=None) for node in edge
        ]
        
        samples.append(edge_list)
        
    return np.array(samples)


import numpy as np

def group_bitarrays_no_overflow(data, task_length):
    """
    Groups bitarrays by sorted operand pairs without integer conversion.

    Parameters:
    - data: np.ndarray of shape (N, 2 * task_length), dtype=bool or similar
    - task_length: int, length of each operand

    Returns:
    - unique_keys: np.ndarray of unique group keys
    - group_indices: np.ndarray of group indices for each bitarray
    """
    # Split data into operands a and b
    a = data[:, :task_length]
    b = data[:, task_length:]

    # Pack the bits into bytes for both operands
    # Ensure that task_length is a multiple of 8 for packbits; pad if necessary
    pad_length = (8 - (task_length % 8)) % 8
    if pad_length > 0:
        a_padded = np.pad(a, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)
        b_padded = np.pad(b, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)
    else:
        a_padded = a
        b_padded = b

    a_bytes = np.packbits(a_padded, axis=1)
    b_bytes = np.packbits(b_padded, axis=1)

    # Define a fixed-length byte string dtype for lexicographical comparison
    byte_length = a_bytes.shape[1]
    dtype = f'S{byte_length}'

    # View the byte arrays as fixed-length byte strings
    a_view = a_bytes.view(dtype).reshape(-1)
    b_view = b_bytes.view(dtype).reshape(-1)

    # Create a mask where a < b lexicographically
    mask = a_view < b_view

    # Sort operands based on the mask
    sorted_a = np.where(mask[:, np.newaxis], a_bytes, b_bytes)
    sorted_b = np.where(mask[:, np.newaxis], b_bytes, a_bytes)

    # Combine sorted_a and sorted_b into unique keys
    combined_keys = np.hstack([sorted_a, sorted_b])

    # Define a fixed-length byte string dtype for combined keys
    combined_byte_length = combined_keys.shape[1]
    combined_dtype = f'S{combined_byte_length}'

    # View combined_keys as single byte strings for uniqueness
    combined_keys_view = combined_keys.view(combined_dtype).reshape(-1)

    # Find unique groups and get inverse indices
    unique_keys, group_indices = np.unique(combined_keys_view, return_inverse=True)

    return unique_keys, group_indices

def split_dataset_optimized(data, task_length, train_size, test_size, seed):
    """
    Splits the dataset into train and test sets ensuring no conflicting pairs.
    Optimized for speed by allocating test groups first and using vectorized operations.

    Parameters:
    - data: np.ndarray of shape (N, 2 * task_length), dtype=bool or similar
    - task_length: int, length of each operand
    - train_size: int, desired number of samples in the train set
    - test_size: int, desired number of samples in the test set
    - seed: int or None, random seed for reproducibility

    Returns:
    - train_set: np.ndarray of shape (train_size, 2 * task_length)
    - test_set: np.ndarray of shape (test_size, 2 * task_length)
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    start_time = time.time()
    # Group bitarrays
    unique_keys, group_indices = group_bitarrays_no_overflow(data, task_length)
    print(f"Time to group: {time.time() - start_time}")

    
    # Precompute all indices for each group
    sorted_order = np.argsort(group_indices)
    sorted_group_indices = group_indices[sorted_order]
    unique_group_ids, group_start_indices, group_counts = np.unique(
        sorted_group_indices, return_counts=True, return_index=True
    )
    groups_sorted = np.split(sorted_order, group_start_indices[1:])

    # Shuffle the groups
    shuffled_group_ids = rng.permutation(len(groups_sorted))
    
    shuffled_groups = [groups_sorted[i] for i in shuffled_group_ids]

    # Precompute group sizes
    group_sizes = np.array([group.size for group in shuffled_groups])
    cumulative_sizes = np.cumsum(group_sizes)

    # Find the cutoff index where cumulative size <= test_size
    test_cutoff = np.searchsorted(cumulative_sizes, test_size, side='right')

    # Select groups for test set
    test_groups = shuffled_groups[:test_cutoff]
    test_indices = np.concatenate(test_groups) if test_cutoff > 0 else np.array([], dtype=int)

    # Assign the remaining groups to train set
    train_groups = shuffled_groups[test_cutoff:]
    train_indices = np.concatenate(train_groups) if test_cutoff < len(shuffled_groups) else np.array([], dtype=int)

    # Handle slight overfill by including the next group if it exceeds test_size
    test_overfill = int(0.1 * test_size)
    if test_indices.size < test_size and test_cutoff < len(shuffled_groups):
        next_group = shuffled_groups[test_cutoff]
        group_size = next_group.size
        if test_indices.size + group_size <= test_size + test_overfill:
            test_indices = np.concatenate([test_indices, next_group])
            train_groups = shuffled_groups[test_cutoff + 1:]
            train_indices = np.concatenate([train_groups[i] for i in range(len(train_groups))]) if len(train_groups) > 0 else train_indices

    # Trim the arrays to exact desired sizes if necessary
    if test_indices.size > test_size + test_overfill:
        test_indices = test_indices[:test_size]
    if train_indices.size > train_size:
        train_indices = train_indices[:train_size]

    # Create the final train and test sets
    train_set = data[train_indices]
    test_set = data[test_indices]

    return train_set, test_set

def group_bitarrays(data, task_length):
    groups = {}
    for bitarray in data:
        x, y = tuple(bitarray[:task_length]), tuple(bitarray[task_length:])
        key = tuple(sorted([x, y]))
        if key not in groups:
            groups[key] = []
        groups[key].append(bitarray)

    return groups
