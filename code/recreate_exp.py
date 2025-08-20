import json
import subprocess
import os
from pathlib import Path
import argparse

import os

import random
import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets import load_from_disk

from models.transformerpp import ModelArgs, Transformer
from train import GPTTrainingModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils import GRAPH_MAXCUT_CHR_DICT, SEQ_CHR_DICT, get_compute_from_steps, find_output_ranges

import logging
import argparse

from utils import CHR_DICT, COT_CHR_DICT, SORTING_CHR_DICT, GRAPH_CHR_DICT
from utils import get_train_loss_and_steps, determine_critical_batch_size

def extract_dataset_info(config_path):
    """
    Extract dataset information from a training run config file.
    """
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    config = data['config']
    
    # Extract the dataset filepath from config
    dataset_path = config['train_dataset_filepath']
    
    # Parse the dataset path to extract components
    # ./datasets/{task}/{repetitions}/{task_length}/{method}/{ablations}/{int(lr * 1e4)}/{input_reverse}/{output_reverse}/{skip_line}/{model_dim}/{num_layers}/{num_heads}/{data_seed}/{model_seed}
    parts = Path(dataset_path).parts

    return {
        # Direct from config
        'task': config['task'],
        'task_length': config['task_length'],
        'method': config['method'],
        'n_embd': config['n_embd'],
        'n_layer': config['n_layer'],
        'n_head': config['n_head'],
        'lr': config['lr'],
        'compute_budget': config['compute_budget'],
        'batch_size': config['batch_size'],
        'test_batch_size': config['test_batch_size'],
        'dropout': config['dropout'],
        'bias': config['bias'],
        'warmup_ratio': config['warmup_ratio'],
        'max_steps': config['max_steps'],
        'weight_decay': config['weight_decay'],
        'grad_clip': config['grad_clip'],
        'betas': config['betas'],
        'validation_ratio': config['validation_ratio'],
        'precision': config['precision'],
        'mask_idx': config.get('mask_idx', None),
        
        # From path parts
        'repetitions': parts[2],
        'ablations': parts[5],
        'input_reverse': parts[7] == 'True',
        'output_reverse': parts[8] == 'True',
        'skip_line': parts[9],
        'data_seed': int(parts[13]),
        'model_seed': int(parts[14]),
    }

def recreate_dataset(config_path, dataset_root_dir='datasets_interp', train_set_size=10000, val_set_size=1000, test_set_size=1000, dataset_seed=None, train_filter_set_path=None):
    """
    Recreate the dataset using the configuration from a training run file.
    """
    info = extract_dataset_info(config_path)
    
    dataset_seed = info['data_seed'] if dataset_seed is None else dataset_seed

    # Construct the dataset directory path
    dataset_dir = f"./{dataset_root_dir}/{info['task']}/{info['repetitions']}/{info['task_length']}/{info['method']}/{info['ablations']}/{info['compute_budget']}/{int(info['lr'] * 1e4)}/{info['input_reverse']}/{info['output_reverse']}/{info['skip_line']}/{info['n_embd']}/{info['n_layer']}/{info['n_head']}/{dataset_seed}/{info['model_seed']}"
    
    # Ensure the dataset directory exists
    os.makedirs(os.path.dirname(dataset_dir), exist_ok=True)
    
    # Construct the command
    command = (
        f"sage --python3 data.py "
        f"--task_length {info['task_length']} "
        f"--number_of_nodes {info['task_length']} "
        f"--test_set_size {test_set_size} "  # Hardcoded as it's not in config
        f"--val_set_size {val_set_size} "   # Hardcoded as it's not in config
        f"--train_set_size {train_set_size} "   # Hardcoded as it's not in config
        f"--seed {dataset_seed} "
        f"--task {info['task']} "
        f"--model_dim {info['n_embd']} "
        f"--num_layer {info['n_layer']} "
        f"--compute_budget {info['compute_budget'] if train_filter_set_path is None else int(1e9)} "
        f"--method {info['method']} "
        f"--ablations {info['ablations']} "
        f"--skip_line {info['skip_line']} "
        f"--dataset_dir {dataset_dir} "
        f"{'--input_reverse' if info['input_reverse'] else ''} "
        f"{'--output_reverse' if info['output_reverse'] else ''} "
        f"{f'--train_filter_set_path {train_filter_set_path}' if train_filter_set_path else ''} "
        f"--repetitions {info['repetitions']}"
    )
    
    print(f"Executing command:\n{command}")
    
    # Execute the command
    process = subprocess.run(command, shell=True, text=True)
    
    if process.returncode == 0:
        print(f"Dataset successfully recreated in: {dataset_dir}")
    else:
        print("Error creating dataset")

    return dataset_dir, info

def create_args_from_config(config_path, dataset_root_dir='datasets_interp', train_set_size=10000, val_set_size=1000, test_set_size=1000, dataset_seed=None, train_filter_set_path=None):
    """
    Create an argparse.Namespace object from the config file.
    """
    dataset_dir, info = recreate_dataset(config_path, dataset_root_dir=dataset_root_dir, train_set_size=train_set_size, val_set_size=val_set_size, test_set_size=test_set_size, dataset_seed=dataset_seed, train_filter_set_path=train_filter_set_path)
    
    args = argparse.Namespace()

    # Dataset paths
    args.dataset_dir = dataset_dir
    args.train_dataset_filepath = f'{dataset_dir}/train_dataset'
    args.val_dataset_filepath = f'{dataset_dir}/val_dataset'
    args.test_dataset_filepath = f'{dataset_dir}/test_dataset'

    # Model parameters
    args.task_length = info['task_length']
    args.task = info['task']
    args.method = info['method']
    args.seed = info['model_seed']
    args.batch_size = info['batch_size']
    args.test_batch_size = info['test_batch_size']
    args.n_layer = info['n_layer']
    args.n_head = info['n_head']
    args.n_embd = info['n_embd']
    args.dropout = info['dropout']
    args.bias = info['bias']

    # Training parameters
    args.warmup_ratio = info['warmup_ratio']
    args.max_steps = info['max_steps']
    args.compute_budget = info['compute_budget']
    args.lr = info['lr']
    args.weight_decay = info['weight_decay']
    args.grad_clip = info['grad_clip']
    args.betas = info['betas']
    args.validation_ratio = info['validation_ratio']
    args.precision = info['precision']
    
    # Optional parameters
    if info['mask_idx'] is not None:
        args.mask_idx = info['mask_idx']

    return args

class ValidateEndCallback(Callback):
    def __init__(self, val_dataloader):
        super(ValidateEndCallback).__init__()
        self.val_dataloader = val_dataloader

    def on_train_end(self, trainer, pl_module):
        trainer.validate(model=pl_module, dataloaders=self.val_dataloader)

class ComprehensiveLogger(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.metrics_log = []
        
        # Create directory structure
        self.checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        self.attention_dir = os.path.join(log_dir, 'attention_scores')
        self.feed_forward_dir = os.path.join(log_dir, 'feed_forward_scores')
        self.residual_activations_dir = os.path.join(log_dir, 'residual_activations')
        self.metrics_file = os.path.join(log_dir, 'metrics.json')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.attention_dir, exist_ok=True)
        os.makedirs(self.feed_forward_dir, exist_ok=True)
        os.makedirs(self.residual_activations_dir, exist_ok=True)
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Collect attention scores from all layers
        for layer_idx, layer in enumerate(pl_module.model.layers):
            if hasattr(layer.attention, 'store_scores') and layer.attention.store_scores:
                scores_dir = os.path.join(
                    self.attention_dir, 
                    f'step_{trainer.global_step}',
                    f'batch_{batch_idx}'
                )
                os.makedirs(scores_dir, exist_ok=True)
                
                # Save all accumulated scores
                for score_idx, score in enumerate(layer.attention.store_scores):
                    np.save(
                        os.path.join(scores_dir, f'layer_{layer_idx}_score_{score_idx}.npy'),
                        score
                    )
                
                # Clear scores after saving
                layer.attention.clear_scores()

            if hasattr(layer.feed_forward, 'store_scores') and layer.feed_forward.store_scores:
                scores_dir = os.path.join(
                    self.feed_forward_dir, 
                    f'step_{trainer.global_step}',
                    f'batch_{batch_idx}'
                )
                os.makedirs(scores_dir, exist_ok=True)
                
                # Save all accumulated scores
                for score_idx, score in enumerate(layer.feed_forward.store_scores):
                    np.save(
                        os.path.join(scores_dir, f'layer_{layer_idx}_score_{score_idx}.npy'),
                        score
                    )
                
                # Clear scores after saving
                layer.feed_forward.clear_scores()

            if hasattr(layer, 'store_residual_activations') and layer.store_residual_activations:
                activations_dir = os.path.join(
                    self.residual_activations_dir, 
                    f'step_{trainer.global_step}',
                    f'batch_{batch_idx}'
                )
                os.makedirs(activations_dir, exist_ok=True)

                for activation_idx, activation in enumerate(layer.store_residual_activations):
                    np.save(
                        os.path.join(activations_dir, f'layer_{layer_idx}_activation_{activation_idx}.npy'),
                        activation
                    )
                
                # Clear activations after saving
                layer.clear_residual_activations()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Collect metrics
        metrics = trainer.callback_metrics
        metrics_dict = {
            'step': trainer.global_step,
            'epoch': trainer.current_epoch,
            'metrics': {k: float(v) for k, v in metrics.items() if torch.is_tensor(v)}
        }
        self.metrics_log.append(metrics_dict)
        
        # Save metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)

class CheckpointEveryNSteps(Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

def train_from_config_filepath(filepath, parent_logdir, other_callbacks=[]):
    args = create_args_from_config(filepath)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    log_dir = os.path.join(parent_logdir, args.task, str(args.task_length), args.method, str(args.compute_budget))  # Replace "run_name" with your experiment name
    
    if os.path.exists(log_dir):
        print(f"Skipping training {args.compute_budget} as {log_dir} already exists")
        return
    
    os.makedirs(log_dir)

    train_dataset = load_from_disk(args.train_dataset_filepath)
    val_dataset = load_from_disk(args.val_dataset_filepath)
    test_dataset = load_from_disk(args.test_dataset_filepath)


    val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size if 'graph' not in args.task else 1, shuffle=False
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    print(train_dataset[0]['ids'])

    block_size = len(val_dataset[0]["ids"])
    sample_tensor = torch.Tensor(val_dataset[0]["ids"])

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

    if args.task in ["addition", "multiplication"]:
        repetitions = len((sample_tensor == 4).nonzero(as_tuple=True)[0])
    else:
        repetitions = 1

    transformerpp_config = ModelArgs(
        max_seq_len=block_size,
        vocab_size=len(tokens_dict.keys()),
        n_layers=args.n_layer,
        n_heads=args.n_head,
        dim=args.n_embd,
        max_batch_size=args.test_batch_size,
        multiple_of=2,
        task=args.task,
        norm_eps=0
    )

    num_batches_per_epoch = len(train_dataset) // args.batch_size
    if len(train_dataset) % args.batch_size != 0:
        num_batches_per_epoch += 1

    total_steps = args.max_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    val_check_interval = max(1, int(total_steps * args.validation_ratio))

    # model = GPT(config)
    base_model = Transformer(transformerpp_config)

    model = GPTTrainingModel(
        base_model,
        warmup_steps,
        total_steps,
        args.lr,
        args.task_length,
        args.betas,
        args.weight_decay,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        repetitions=repetitions,
        method=args.method,
        log_activations=False
    )

    # Initialize callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # comprehensive_logger = ComprehensiveLogger(log_dir=log_dir)
    # checkpoint_callback = ModelCheckpoint(dirpath=f"{log_dir}/checkpoints", every_n_train_steps=val_check_interval, every_n_epochs=None, train_time_interval=None)
    checkpoint_callback = CheckpointEveryNSteps(val_check_interval)
    callback_list = [ValidateEndCallback(val_dataloader), lr_monitor, checkpoint_callback]

    for c in other_callbacks:
        callback_list.append(c)

    trainer = Trainer(
        max_steps=args.max_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        callbacks=callback_list,
        default_root_dir=log_dir,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, required=True, help='Directory containing experiment results')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Directory to save training logs')
    parser.add_argument('--task_length', type=str, required=True, help='Task length to filter for')
    parser.add_argument('--max_compute_exponent', type=float, default=float('inf'), help='Maximum compute budget to consider')
   
    args = parser.parse_args()
    args.max_compute = 10 ** args.max_compute_exponent

    # Walk through individual runs directory
    individual_runs_dir = os.path.join(args.directory, args.task_length, 'individual_runs')
    if not os.path.isdir(individual_runs_dir):
        raise ValueError(f"Individual runs directory not found: {individual_runs_dir}")

    # Process each compute budget directory
    compute_dirs = []
    for compute_dir in os.listdir(individual_runs_dir):
        try:
            compute_budget = int(compute_dir)
            if compute_budget <= args.max_compute:
                compute_dirs.append((compute_budget, compute_dir))
        except ValueError:
            print(f"Skipping non-integer compute budget directory: {compute_dir}")
            continue
    
    # Sort by compute budget to process in order
    compute_dirs.sort()

    for compute_budget, compute_dir in compute_dirs:
        print(f"Processing compute budget: {compute_budget}")
        compute_path = os.path.join(individual_runs_dir, compute_dir)
        if not os.path.isdir(compute_path):
            continue

        # Find file with minimum validation loss
        min_val_loss = float('inf')
        min_loss_filepath = None

        for file in os.listdir(compute_path):
            if not file.endswith('.json'):
                continue

            file_path = os.path.join(compute_path, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                val_loss = data.get('summary', {}).get('val_loss')
                if val_loss is not None and val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_loss_filepath = file_path
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {file_path}")
                continue

        if min_loss_filepath:
            print(f"\nTraining model for compute budget {compute_budget} using config from: {min_loss_filepath}")
            try:
                train_from_config_filepath(min_loss_filepath, args.ckpt_dir)
            except Exception as e:
                print(f"Error training model: {e}")
        else:
            print(f"No valid configuration file found for compute budget: {compute_budget}")

if __name__ == "__main__":
    main()