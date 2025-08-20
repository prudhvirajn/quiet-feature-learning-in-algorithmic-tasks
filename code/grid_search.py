import random

from calculate_max_samples import get_ctx, get_examples_from_compute
import typer # type: ignore
import pandas as pd
from job_scheduler import scheduling_runs

import os
import csv
import time
from collections import defaultdict

from utils import *

import numpy as np
import wandb
import json
import math

import logging
import sys
import itertools

from typing import Tuple

# Define the columns for the DataFrame

api = wandb.Api()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # Set the stream to sys.stdout
)

logger = logging.getLogger(__name__)


def nearest_power_of_two(num):
    return 2 ** math.floor(math.log2(num))


def scrape_runs(project_name: str):
    runs = None

    for _ in range(1):
        try:
            runs = api.runs(
                project_name, order="-created_at"
            )  # Order by creation time, descending
            runs = list(runs)
            break
        except TypeError as error:
            time.sleep(10)
            continue

    train_stats = {}
    for idx, run in enumerate(runs):
        train_stats[idx] = {}
        train_stats[idx]["config"] = run.config
        train_stats[idx]["summary"] = run.summary._json_dict

    return train_stats


def generate_jobs(
    experiment: str = "run_experiments.sh",
    task: str = "addition",
    seed_random: bool = True,
    loss_denom: int = 4,
    precision: int | str = 32,
    repeat_each_task: int = 1,
    hyperparam_combinations: list[Tuple[int, float, int, int]] = [],
    task_length: int = 10,
    number_of_nodes: int = 10,
    method: str = None,
    ablations: str = None,
    num_heads: int = 4,
    iteration_ids: str = "1099",
    test_batch_size: int = 128,
    compute_budget: int = None,
    test_set_size: int = 1000,
    val_set_size: int = 5000,
    data_seed=[],
    model_seed=[],
    input_reverse: bool = True,
    output_reverse: bool = True,
    skip_line: int = 1,
    early_stopping: bool = False,
    threshold_testing: bool = False,
    repetitions: int = 1,
    stats_dirpath: str = None,
):
    # ########################
    # Generate df
    # ########################
    columns = [
        "experiment",
        "task_length",
        "number_of_nodes",
        "method",
        "ablations",
        "model_dim",
        "num_layers",
        "num_heads",
        "task",
        "repetitions",
        "lr",
        "iteration_ids",
        "train_batch_size",
        "test_batch_size",
        # "max_num_samples",
        "est_compute",
        "test_set_size",
        "val_set_size",
        "data_seed",
        "model_seed",
        "loss_denom",
        "precision",
        "input_reverse",
        "output_reverse",
        "skip_line",
        "early_stopping",
        "threshold_testing",
    ]
    df = pd.DataFrame(columns=columns)
    # ########################
    # Append df
    # ########################
    index_counter = 0
    # ctx = get_ctx(task, task_length, method, ablations, skip_line)
    for train_batch_size, lr, model_dim, num_layers in hyperparam_combinations:
        for ind_rep in range(repeat_each_task):
            # max_samples = get_examples_from_compute(
            #     model_dim, ctx, compute_budget, n_layers=num_layer
            # )

            if seed_random:
                data_seed = random.randint(0, 40000)
                model_seed = random.randint(0, 40000)
            else:
                data_seed = data_seed[index_counter]
                model_seed = model_seed[index_counter]

            row_to_add = {
                "experiment": experiment,
                "task_length": task_length,  # Example values, replace with your actual data
                "number_of_nodes": number_of_nodes,
                "method": method,
                "ablations": ablations,
                "model_dim": model_dim,
                "num_layers": int(num_layers),
                "num_heads": int(num_heads),
                "task": task,
                "repetitions": repetitions,
                "lr": lr,
                "iteration_ids": iteration_ids,
                "train_batch_size": train_batch_size,
                "test_batch_size": test_batch_size,
                # "max_num_samples": int(max_samples),
                "est_compute": int(compute_budget),
                "test_set_size": int(test_set_size),
                "val_set_size": int(val_set_size),
                "data_seed": data_seed,
                "model_seed": model_seed,
                "loss_denom": loss_denom,
                "precision": precision,
                "input_reverse": input_reverse,
                "output_reverse": output_reverse,
                "skip_line": skip_line,
                "early_stopping": early_stopping,
                "threshold_testing": threshold_testing,
                "stats_dirpath": stats_dirpath,
            }
            # print (row_to_add)
            df_to_add = pd.DataFrame(row_to_add, index=[0])
            df = pd.concat([df, df_to_add], ignore_index=True)

            index_counter += 1

    return df


def get_precision(task_length: int):
    return 32
    # if task_length < 30:
    #     return 32
    # else:
    #     return "bf16-mixed"

def filter_by_steps(train_batch_size, model_dim, num_layer, ctx, compute_budget, min_steps=100, max_steps=int(1e5)):
    steps = get_steps_from_compute(
        model_dim,
        ctx,
        compute_budget,
        train_batch_size,
        n_layers=num_layer,
    )
    
    return min_steps <= steps <= max_steps


def main(
    task: str = "addition",
    task_length: int = 10,
    number_of_nodes: int = 10,
    method: str = "normal",
    ablations: str = "11111",
    train_batch_sizes: list[int] = [],
    test_batch_size: int = 128,
    compute_budget: int = int(1e6),
    min_steps: int = 1,
    max_steps: int = int(1e5),
    model_dims: list[int] = [],
    learning_rates: list[float] = [1e-1, 1e-2, 1e-3, 1e-4],
    num_layers: list[int] = [],
    num_heads: int = 4,
    iteration_id: int = 1000,
    loss_denom: int = 4,
    input_reverse: bool = False,
    output_reverse: bool = False,
    skip_line: int = 1,
    results_dir: str = "results",
    repetitions: int = 1,
):
    stats_dirpath = f"{results_dir}/{task}/{repetitions}/{method}/{ablations}/{loss_denom}/input_reverse{input_reverse}/output_reverse{output_reverse}/{task_length}"

    individual_runs_dirpath = os.path.join(stats_dirpath, "individual_runs")
    os.makedirs(individual_runs_dirpath, exist_ok=True)

    if "graph" not in task:
        ctx = get_ctx(task, task_length, method, ablations, skip_line, number_of_nodes = number_of_nodes)
    else:
        if task == "graph_path":
            CTX_DICT = {8: 35.5, 10: 51.9, 11: 61.9140625}
        elif task in ["graph_breadth_first_search", "graph_depth_first_search", "graph_topological_sort"]:
            CTX_DICT = {8: 39.984375, 9: 48.625, 10: 58.4375, 11: 69.359375}
        elif task == "graph_min_spanning_tree_kruskal":
            CTX_DICT = {10: 65.4375, 11: 77.359375}
        elif task == "graph_longest_path":
            CTX_DICT = {10: 57.40625, 11: 68.359375}
        
        ctx = CTX_DICT[number_of_nodes]

    print("Generating hp combinations")

    hyperparam_combinations = list(
        itertools.product(train_batch_sizes, learning_rates, model_dims, num_layers)
    )

    hyperparam_combinations = [i for i in hyperparam_combinations if (filter_by_steps(i[0], i[2], i[3], ctx, compute_budget, min_steps, max_steps))]

    if not len(hyperparam_combinations): raise Exception("Could not find unique hyperparameter triplets")

    random.shuffle(hyperparam_combinations)

    uncompleted_hyperparameter_combinations = []

    run_history_dir = os.path.join(individual_runs_dirpath, str(compute_budget))

    if os.path.exists(run_history_dir):
        completed_run_files = os.listdir(os.path.join(individual_runs_dirpath, str(compute_budget)))

        for train_bs, lr, n_embd, n_layer in hyperparam_combinations:
            filename = f"results_dim{n_embd}_layer{n_layer}_head{num_heads}_lr{int(lr * 1e4)}_batchsize{train_bs}.json"

            if filename not in completed_run_files:
                uncompleted_hyperparameter_combinations.append([train_bs, lr, n_embd, n_layer])
    else:
        uncompleted_hyperparameter_combinations = hyperparam_combinations
    
    jobs_df = generate_jobs(
        task=task,
        task_length=task_length,
        number_of_nodes=number_of_nodes,
        method=method,
        ablations=ablations,
        num_heads=num_heads,
        hyperparam_combinations=uncompleted_hyperparameter_combinations,
        test_batch_size=test_batch_size,
        compute_budget=compute_budget,
        test_set_size=1000 if 2 ** (task_length * 2) > 2000 else 64,
        val_set_size=1000 if 2 ** (task_length * 2) > 2000 else 64,
        iteration_ids=iteration_id,
        loss_denom=loss_denom,
        input_reverse=input_reverse,
        output_reverse=output_reverse,
        skip_line=skip_line,
        repetitions=repetitions,
        precision=get_precision(task_length),
        stats_dirpath=stats_dirpath,
    )

    print("Jobs defined")

    scheduling_runs(
        jobs_df,
        "5",
        mode="standard",
        debug=False,
    )

    completed_run_files = os.listdir(os.path.join(individual_runs_dirpath, str(compute_budget)))
    num_runs = len(hyperparam_combinations)
    num_completed = len(completed_run_files)

    print(num_runs)

    assert num_runs <= num_completed, f"Some runs have failed to complete: {num_completed} / {num_runs}"


if __name__ == "__main__":
    typer.run(main)
