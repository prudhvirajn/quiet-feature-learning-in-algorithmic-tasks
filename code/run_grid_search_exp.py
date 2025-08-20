from grid_search import main as grid_search
import json
import itertools
import random
import concurrent.futures
import argparse

import random
from math import comb

from multiprocessing import Pool, cpu_count


def generate_ablation_combinations():
    return ["".join(bits) for bits in itertools.product("01", repeat=5)]


def run_experiment_wrapper(args_dict, retry=1):
    try:
        grid_search(**args_dict)
    except Exception as error:
        print(f"Run failed: {args_dict} because of {error}")


def handle_error(exception, config=None):
    # Improved error handling code to print the configuration if provided
    error_message = f"Error: {exception}"
    if config is not None:
        error_message += f"\nConfiguration: {config}"
    print(error_message)
    
def load_config_from_file(config_file_path):
    """Load configuration from a JSON file."""
    with open(config_file_path, 'r') as f:
        return json.load(f)


def generate_configs_for_task(task_config, results_dir, iteration_id_start):
    """Generate all hyperparameter combinations for a single task."""
    configs = []
    iteration_id = iteration_id_start
    
    # Extract hyperparameter ranges
    task = task_config["task"]
    task_lengths = task_config["task_lengths"]
    compute_budget_exponents = task_config["compute_budget_exponents"]
    train_batch_sizes = task_config["train_batch_sizes"]
    test_batch_size = task_config.get("test_batch_size", 64)
    model_dims = task_config["model_dims"]
    learning_rates = task_config["learning_rates"]
    num_layers = task_config["num_layers"]
    num_heads = task_config.get("num_heads", 4)
    loss_denom = task_config.get("loss_denom", 4)
    skip_line = task_config.get("skip_line", 1)
    repetitions = task_config.get("repetitions", 1)
    input_reverse = task_config.get("input_reverse", True)
    output_reverse = task_config.get("output_reverse", True)

    for compute_budget_exponent in compute_budget_exponents:
        iteration_id += 10
        for idx, task_length in enumerate(task_lengths):
            number_of_nodes = task_length
            config = {
                "task": task,
                "task_length": task_length,
                "method": "normal",
                "ablations": "11111",
                "train_batch_sizes": train_batch_sizes,
                "test_batch_size": test_batch_size,
                "compute_budget": int(10 ** compute_budget_exponent),
                "model_dims": model_dims,
                "learning_rates": learning_rates,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "iteration_id": iteration_id,
                "loss_denom": loss_denom,
                "input_reverse": input_reverse,
                "output_reverse": output_reverse,
                "skip_line": skip_line,
                "results_dir": results_dir,
                "repetitions": repetitions,
                "number_of_nodes": number_of_nodes,
            }
            configs.append(config)
    
    return configs, iteration_id


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter experiments from config file')
    parser.add_argument('--config', required=True, help='Path to the configuration JSON file')
    parser.add_argument('--max_workers', type=int, default=8, help='Maximum number of worker processes')
    args = parser.parse_args()
    
    # Load configuration from file
    config_data = load_config_from_file(args.config)
    
    # Extract global settings
    results_dir = config_data.get("results_dir", "results")
    iteration_id_start = config_data.get("iteration_id_start", 1000)
    
    # Generate & Run all configs for all tasks
    for task_config in config_data["tasks"]:
        all_configs = []
        current_iteration_id = iteration_id_start
    
        task_configs, current_iteration_id = generate_configs_for_task(
            task_config, results_dir, current_iteration_id
        )
        all_configs.extend(task_configs)
    
        print(f"Generated {len(all_configs)} total configurations")
        print("Launching configs")
        random.shuffle(all_configs)
        
        # Use ProcessPoolExecutor to run the experiments concurrently
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit the initial batch of tasks
            future_to_config = {
                executor.submit(run_experiment_wrapper, config): config
                for config in all_configs
            }

            for future in concurrent.futures.as_completed(future_to_config):
                # Retrieve the result
                result = future.result()
                config = future_to_config[future]


if __name__ == "__main__":
    main()
