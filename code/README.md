## Overview

This code recreates scaling law and feature probe results from the main paper. There are three stages to recreate the results:

1. Run scaling law experiment for tasks and input sizes. (Refer to "Running scaling laws experiments" section)

2. Train Feature Probes to detect Quiet / Loud Features. (Refer to "Training Feature Probes to detect Quiet / Loud Features across compute budget" section)

3. Causal Intervention using feature probes to determine if a feature is causally necessary to the model's performance. (Refer to "Feature Ablations" section)

## Installation

1. Install [mamba](https://github.com/mamba-org/mamba)
2. Run the following command to install dependencies:

```
mamba create -n sage sage datasets lightning scikit-learn tokenizers torch typer wandb python=3.10
```

3. Activate environment `mamba activate sage`
4. Run wandb init to initialize wandb. Select the entity where you want to record logs. The project name is irrelevant so feel free to use any project name.
5. For graph tasks, install Nauty and Traces: https://pallini.di.uniroma1.it/, then generate graphs by running `geng -c 11 > graphs/graph11c.g6` or download 11 vertex simple graph at http://users.cecs.anu.edu.au/~bdm/data/graphs.html

## Running scaling laws experiments

The following command runs grid search for tasks and inputs sizes. The command will create datasets and train models appropriately. After each training run finishes, it records the final train / test metrics in a json file.

Previous results are provided in `data/scaling_law_results` in case you want to skip the step.

```
python3 run_grid_search_exp.py --config "path_to_config.json"
```

For non-graph tasks: `scaling_law_config_nongraph_tasks.json`

For graph tasks: `scaling_law_config_graph_tasks.json`

For custom config files, they should be structured as follows:

```
{
  "results_dir": "directory-path-to-store-results",
    "iteration_id_start": "number-for-unique-identification-of-wandb-project",
    "tasks": [
        {
            "task": "task",
            "task_lengths": "input-sizes",
            "compute_budget_exponents": "list of compute budget exponents, for ex: [9, 10, 11, 12, 13, 14, 15]",
            "train_batch_sizes": "list of train batch sizes",
            "test_batch_size": "test batch size",
            "model_dims": "list of transformer model dimension, for ex: [8, 64, 512]",
            "learning_rates": "list of learning rates, for ex: [1e-1, 1e-2]",
            "num_layers": "list of number of layers, for ex: [4, 16] ",
            "num_heads": "number of heads",
            "loss_denom": 4 (irrelevant paramter, leave it at 4),
            "skip_line": 1 (irrelevant paramter, leave it at 1),
            "repetitions": 1 (irrelevant paramter, leave it at 1),
            "input_reverse": true (whether to reverse input, least-to-most significant, for binary tasks),
            "output_reverse": true (whether to reverse output, least-to-most significant, for binary tasks),
            "method": "normal" (irrelevant paramter, leave it at "normal"),
            "ablations": "11111" (irrelevant paramter, leave it at "11111")
        },
}
```

## Plotting scaling law

To plot scaling law results, run

```
python3 plot_scaling_law.py "results-directory-path" "output-directory-path"
```

"results-directory-path": Previous results are given in `data/scaling_law_results`

## Training Feature Probes to detect Quiet / Loud Features across compute budget

To compute feature learning for a specific task across compute budgets, run

```
python3 train_feature_probes_across_compute.py \
  --directory "results-task-specific-directory-path" \ 
  --ckpt_dir "directory-path-to-save-checkpoints" \
  --task_length "input-size" \
  --max_compute_exponent "max-compute-exponent" \
  --target_budget_exponent "target-compute-budget-exponent" \
  --output_pickle "output-pickle-path \
  --do_train"
```

`results-task-specific-directory-path`: Task directory where scaling law results are stored. For example: `data/scaling_law_results`

`target-compute-budget-exponent`: Exponent for the compute budget, based on which the model size is chosen. The compute-optimal model size for this compute budget will be chosen. For example, if target compute budget is $10^{16}$ then 16 is the compute budget exponent.

`max-compute-exponent`: Exponent for the largest compute budget to train the fixed-model size. The fixed model size chosen previously, is trained from 10^9 to 10^{max-compute-exponent}. The max-compute-exponent is 15 for all tasks except for multiplication for which it is 16.

`--do_train`: Trains fixed model-size (model size is the compute optimal model for target compute budget)

For results from the paper, refer to `data/feature_probe_results_across_compute_budgets` 

## Training Feature Probes to detect Quiet / Loud Features within a single training run

To compute feature learning for a specific task across compute budgets, run

```
python3 train_feature_probes_within_single_training_run.py \
  --directory "results-task-specific-directory-path" \
  --ckpt_dir "directory-path-to-save-checkpoints" \
  --task_length "input-size" \
  --max_compute_exponent "max-compute-exponent" \
  --target_budget_exponent "target-compute-budget-exponent" \
  --output_pickle "output-pickle-path \
  --do_train"
```

## Plotting Quiet / Loud Features across compute budgets

To plot quiet / loud features, run:

```
python3 plot_quiet_loud_features.py data/feature_probe_results_across_compute_budgets "plot-output-dir"
```

## Feature Ablations

To calculate causal effect of features, we ablate features and measure test accuracy. Run the following command:

```
python3 feature_ablations.py \
  --directory "results-task-specific-directory-path" \
  --pickle_file "path-to-pickle-file" \
  --ckpt_dir "directory-path-to-save-checkpoints" \
  --task "task" \
  --task_length "input-size" \
  --metric train_log_loss \
  --label_name "feature-name" \
  --target_budget_exponent "compute-budget-exponent" \
  --output_json "output-results-json-filepath"
```

## License Information

Our Transformer++ implementation is based on Meta's Llama 2 model architecture. The original Llama 2 
is licensed under the LLAMA 2 Community License (see LICENSE file). 

Our modifications include:
- Added cross-entropy loss for training
- Added support for feature ablations
- Replaced Grouped Query Attention, with normal attention.
- Removed KV cache optimization.
- Model Arguments is determined by grid-search rather than default LLAMA model arguments.