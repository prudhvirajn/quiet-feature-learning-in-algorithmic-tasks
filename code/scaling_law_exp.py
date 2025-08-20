import argparse
import logging
import os
import shutil
import subprocess
import json
import wandb
import uuid

from datasets import load_from_disk
import numpy as np

from utils import (
    find_first_token_instance,
    get_steps_from_compute,
    get_compute_from_steps,
    get_max_loss,
    get_train_loss_and_steps,
    get_compute,
)

NUM_GPUs = 1


def generate_args(
    lr,
    dim,
    n_layer,
    n_head,
    train_batch_size,
    test_batch_size,
    threshold,
    early_stopping,
    threshold_testing,
    compute,
    average_context_size,
    block_size,
    project_name,
    task_length,
    task,
    method,
    train_dataset_filepath,
    val_dataset_filepath,
    test_dataset_filepath,
    seed,
    precision,
    log_gradients=False,
):
    args = argparse.Namespace()
    args.train_dataset_filepath = train_dataset_filepath
    args.val_dataset_filepath = val_dataset_filepath
    args.test_dataset_filepath = test_dataset_filepath
    args.task_length = task_length
    args.task = task
    args.method = method
    args.seed = seed
    args.n_layer = n_layer
    args.n_head = n_head
    args.n_embd = dim
    args.dropout = 0
    args.bias = False
    args.warmup_ratio = 0.1
    args.validation_ratio = 0.005
    args.batch_size = train_batch_size
    args.test_batch_size = test_batch_size
    args.compute_budget = compute
    args.max_steps = int(
        get_steps_from_compute(
            dim,
            average_context_size,
            compute,
            args.batch_size * NUM_GPUs,
            n_layers=n_layer,
        )
    )
    print(f"Max Steps: {args.max_steps}")
    print(average_context_size, args.max_steps)

    args.project_name = project_name
    args.run_id = str(uuid.uuid4())
    args.lr = lr
    args.loss_threshold = threshold
    args.early_stopping = early_stopping
    args.threshold_testing = threshold_testing
    args.mask_idx = 2 * args.task_length + 1
    args.precision = precision
    args.log_gradients = log_gradients
    return args


def save_run_results(stats_dirpath, args, run_config, run_summary):
    individual_runs_dirpath = os.path.join(stats_dirpath, "individual_runs", f"{int(args.compute_budget)}")
    os.makedirs(individual_runs_dirpath, exist_ok=True)
    
    filename = f"results_dim{args.n_embd}_layer{args.n_layer}_head{args.n_head}_lr{int(args.lr * 1e4)}_batchsize{args.batch_size}.json"
    filepath = os.path.join(individual_runs_dirpath, filename)
    
    result = {
        "config": run_config,
        "summary": run_summary
    }
    
    with open(filepath, 'w') as f:
        json.dump(result, f)
    
    print(f"Results saved to {filepath}")

def check_existing_run(stats_dirpath, args):
    individual_runs_dirpath = os.path.join(stats_dirpath, "individual_runs", f"{int(args.compute_budget)}")

    if not os.path.exists(individual_runs_dirpath):
        return False

    pattern = f"results_dim{args.n_embd}_layer{args.n_layer}_head{args.n_head}_lr{int(args.lr * 1e4)}_batchsize{args.batch_size}.json"
    
    matching_files = [f for f in os.listdir(individual_runs_dirpath) if f == pattern]
    
    return len(matching_files) > 0

def train_for_best_lr(
    model_dim,
    n_layer,
    n_head,
    train_batch_size,
    test_batch_size,
    loss_threshold,
    early_stopping,
    threshold_testing,
    compute,
    average_context_size,
    block_size,
    train_dataset_filepath,
    val_dataset_filepath,
    test_dataset_filepath,
    task,
    method,
    task_length,
    initial_lr_values,
    entity,
    project_name,
    manager_logger,
    seed,
    precision,
    stats_dirpath,
    log_gradients=False,
):
    for lr in initial_lr_values:
        args = generate_args(
            lr,
            model_dim,
            n_layer,
            n_head,
            train_batch_size,
            test_batch_size,
            loss_threshold,
            early_stopping,
            threshold_testing,
            compute,
            average_context_size,
            block_size,
            project_name,
            task_length,
            task,
            method,
            train_dataset_filepath,
            val_dataset_filepath,
            test_dataset_filepath,
            seed,
            precision,
            log_gradients=log_gradients,
        )
        
        print("Checking if this configuration has been run before")
        # Check if this configuration has been run before
        if check_existing_run(stats_dirpath, args):
            print(f"Skipping existing configuration: dim{args.n_embd}_layer{args.n_layer}_head{args.n_head}_lr{args.lr}")
            continue
        
        # Train the model using these args
        args_dict = vars(args)
        cmd_args = []

        for key, value in args_dict.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        cmd_args.append(f"--{key}")
                else:
                    cmd_args.extend([f"--{key}", str(value)])

        cmd = ["python3", "main.py"] + cmd_args
        print(cmd)
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            stdout_output = result.stdout
        except subprocess.CalledProcessError as e:
            print("Error occurred:")
            print(e.stderr)
            continue

        # Fetch the run data from wandb
        api = wandb.Api()
        run = api.run(f"{entity}/{project_name}/{args.run_id}")
        
        # Save the results
        save_run_results(stats_dirpath, args, run.config, run.summary._json_dict)

    manager_logger.info(
        f"Completed training for Loss: {loss_threshold}, Model Dimension: {model_dim}, Batch Size: {train_batch_size * NUM_GPUs}"
    )


# Main management script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ML models with different arguments for binary addition tasks."
    )

    parser.add_argument("--task", default="addition", help="Task to use for training.")
    parser.add_argument("--method", default="cot", help="Method to use for training.")
    parser.add_argument("--ablations", default="11111", help="Ablations")
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        default=False,
        help="Flag for early stopping",
    )
    parser.add_argument(
        "--threshold_testing",
        action="store_true",
        default=False,
        help="Flag for testing at different loss thresholds",
    )
    parser.add_argument(
        "--task_length",
        type=int,
        default=10,
        help="Bit length for the binary addition task.",
    )
    parser.add_argument(
        "--train_dataset_filepath",
        required=True,
        help="File path for the training dataset.",
    )
    parser.add_argument(
        "--val_dataset_filepath",
        required=True,
        help="File path for the validation dataset.",
    )
    parser.add_argument(
        "--test_dataset_filepath", required=True, help="File path for the test dataset."
    )
    parser.add_argument(
        "--loss_denom", default=1, type=float, help="Fraction of max loss"
    )
    parser.add_argument(
        "--learning_rates",
        nargs="+",
        type=float,
        default=[1e-2],
        help="Learning rates to use.",
    )
    parser.add_argument(
        "--log_gradients",
        action="store_true",
        default=False,
        help="Flag to log gradients.",
    )
    parser.add_argument(
        "--training_type",
        default="normal",
        choices=["normal", "critical"],
        help="Type of training to perform.",
    )
    parser.add_argument(
        "--initial_compute_estimate",
        type=int,
        default=5300312000000,
        help="Initial compute estimate.",
    )
    parser.add_argument("--seed", type=int, help="Seed for random number generation.")
    parser.add_argument("--train_batch_size", type=int, default=2048)
    parser.add_argument("--test_batch_size_scaling", type=int, default=2048)
    parser.add_argument("--precision", default="32", type=str)
    parser.add_argument(
        "--model_dimensions",
        nargs="+",
        type=int,
        default=[24, 48, 72, 96],
        help="List of model dimensions to use.",
    )
    parser.add_argument("--num_layer", type=int, default=6)
    parser.add_argument("--num_head", type=int, default=6)
    parser.add_argument(
        "--critical_batch_sizes",
        nargs="+",
        required=False,
        help="List of model dimensions to use.",
    )
    parser.add_argument(
        "--iteration_ids",
        nargs="+",
        type=int,
        default=[11, 12, 13],
        help="iteration id",
    )
    parser.add_argument(
        "--repetitions", type=int, default=1, help="# of examples per sample"
    )
    parser.add_argument(
        "--stats_dirpath",
        required=True,
        help="Directory path for saving individual run results.",
    )
    args = parser.parse_args()

    entity = wandb.Api().default_entity
    method = args.method
    ablations = args.ablations
    task = args.task
    repetitions = args.repetitions
    task_length = args.task_length
    seed = args.seed

    train_dataset_filepath = args.train_dataset_filepath
    val_dataset_filepath = args.val_dataset_filepath
    test_dataset_filepath = args.test_dataset_filepath

    model_dimensions = args.model_dimensions
    n_layer = args.num_layer
    n_head = args.num_head
    learning_rates = args.learning_rates  
    loss_denom = args.loss_denom

    log_gradients = args.log_gradients
    training_type = args.training_type
    precision = args.precision
    early_stopping = args.early_stopping
    threshold_testing = args.threshold_testing

    train_batch_size = args.train_batch_size

    normal_batch_size_dict = {
        model_dim: int(args.test_batch_size_scaling * model_dimensions[-1] / model_dim)
        for model_dim in model_dimensions
    }

    if training_type == "critical":
        critical_batch_size_dict = {
            dim: cbs
            for cbs, dim in zip(args.critical_batch_sizes, args.model_dimensions)
        }
    else:
        critical_batch_size_dict = {}

    manager_logger = logging.getLogger("run_scaling_law_exp")
    manager_logger.setLevel(logging.INFO)

    val_dataset = load_from_disk(args.val_dataset_filepath)
    block_size = len(val_dataset[0]["ids"])
    sample_size = min(len(val_dataset), 128)
    average_context_size = np.mean(
        [
            find_first_token_instance(val_dataset[idx]["attention_mask"], end_token=1)
            + np.sum(val_dataset[idx]["attention_mask"])
            for idx in range(sample_size)
        ]
    )
    print(f"Context Window Size: {block_size}")
    max_loss = get_max_loss(task_length, repetitions, block_size)
    loss_thresholds = [max_loss / loss_denom]

    initial_compute_estimate = args.initial_compute_estimate

    min_compute_dict = {
        loss_threshold: initial_compute_estimate for loss_threshold in loss_thresholds
    }

    for idx, loss_threshold in enumerate(loss_thresholds):
        for iteration_id in args.iteration_ids:
            project_name = f"binary-{task}{repetitions}-gridsearch-{task_length}-{method}-{ablations}-iteration-{iteration_id}"

            critical_batch_size_dict = {}
            compute = min_compute_dict[loss_threshold]

            best_compute = compute
            for model_dim in model_dimensions:
                test_batch_size = normal_batch_size_dict[model_dim]

                train_for_best_lr(
                    model_dim,
                    n_layer,
                    n_head,
                    train_batch_size,
                    test_batch_size,
                    loss_threshold,
                    early_stopping,
                    threshold_testing,
                    compute,
                    average_context_size,
                    block_size,
                    train_dataset_filepath,
                    val_dataset_filepath,
                    test_dataset_filepath,
                    task,
                    method,
                    task_length,
                    learning_rates,
                    entity,
                    project_name,
                    manager_logger,
                    seed,
                    precision,
                    args.stats_dirpath,
                    log_gradients=log_gradients,
                )
                # shutil.rmtree(train_dataset_filepath)
                # shutil.rmtree(val_dataset_filepath)
                # shutil.rmtree(test_dataset_filepath)
