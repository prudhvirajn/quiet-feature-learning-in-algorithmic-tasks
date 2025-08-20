import random
import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets import load_from_disk
from models.transformerpp import ModelArgs, Transformer

from train import GPTTrainingModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from utils import get_compute_from_steps, find_output_ranges

import logging
import os
import argparse
import wandb
import json

from utils import CHR_DICT, COT_CHR_DICT, SORTING_CHR_DICT, GRAPH_CHR_DICT, GRAPH_MAXCUT_CHR_DICT, SEQ_CHR_DICT
from utils import get_train_loss_and_steps, determine_critical_batch_size


class ThresholdTestingCallback(Callback):
    def __init__(
        self,
        test_dataloader: DataLoader,
        loss_threshold: float,
        already_tested=set(),
    ):
        super(ThresholdTestingCallback, self).__init__()
        self.test_dataloader = test_dataloader
        self.loss_threshold = loss_threshold
        self.already_tested = already_tested

    def test(self, pl_module):
        test_accuracy_result = 0
        test_accuracy_reasoning = 0
        total_count = 0
        model = pl_module.model

        for batch in self.test_dataloader:
            input_ids = torch.vstack(batch["ids"]).T.to(model.lm_head.weight.device)
            logits, _ = model(
                input_ids,
                targets=input_ids[:, 1:].long(),
                mask_indices=pl_module.mask_indices,
            )

            num_input_tokens = 2 * pl_module.num_bits + 2
            total_tokens = model.config.block_size // pl_module.repetitions

            if "addition" in model.config.task:
                result_tokens = pl_module.num_bits + 2
            elif "multiplication" in model.config.task:
                result_tokens = 2 * pl_module.num_bits + 1

            remaining_tokens = total_tokens - num_input_tokens
            input_tokens = input_ids[:, :num_input_tokens]
            label_ids = input_ids[:, total_tokens - result_tokens : total_tokens]

            outputs = model.generate(
                input_tokens, remaining_tokens, temperature=1e-9, top_k=1
            )
            check = torch.all(
                outputs[:, total_tokens - result_tokens :] == label_ids, 1
            )
            test_accuracy_result += torch.sum(check).item()
            total_count += input_ids.shape[0]

            if "cot" in pl_module.method:
                reasoning_label_ids = input_ids[
                    :, num_input_tokens : total_tokens - result_tokens
                ]
                reasoning_check = torch.all(
                    outputs[:, num_input_tokens : total_tokens - result_tokens]
                    == reasoning_label_ids,
                    1,
                )
                test_accuracy_reasoning += torch.sum(reasoning_check).item()

        metrics = {"test_acc_result": test_accuracy_result / total_count}

        if "cot" in pl_module.method:
            metrics["test_acc_reasoning"] = test_accuracy_reasoning / total_count

        return metrics

    def on_validation_end(self, trainer, pl_module):
        current_val_loss = trainer.callback_metrics.get("val_loss").item()
        thresholds = [
            self.loss_threshold * 16,
            self.loss_threshold * 8,
            self.loss_threshold * 4,
            self.loss_threshold * 2,
            self.loss_threshold,
        ]

        for threshold in thresholds:
            if current_val_loss <= threshold and threshold not in self.already_tested:
                # Run the test and log the results
                metrics = self.test(pl_module)
                trainer.logger.experiment.log(metrics)
                self.already_tested.add(threshold)
                break  # Stop after testing for the first unmet threshold


class ValidateEndCallback(Callback):
    def __init__(self, val_dataloader):
        super(ValidateEndCallback).__init__()
        self.val_dataloader = val_dataloader

    def on_train_end(self, trainer, pl_module):
        trainer.validate(model=pl_module, dataloaders=self.val_dataloader)


class GradientLoggingCallback(Callback):
    def __init__(self):
        super(GradientLoggingCallback).__init__()

        self.logger = logging.getLogger(__name__)

    # @rank_zero_only
    def safe_print(self, *args, **kwargs):
        print(*args, **kwargs)

    def on_after_backward(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return

        # Store gradients
        gradients = []
        for param in pl_module.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
        gradient_vector = torch.cat(gradients)

        # Compute squared norm
        squared_norm = gradient_vector.dot(gradient_vector)

        # Log the squared norm for single-GPU or CPU case to progress bar and print
        self.logger.info(f"{trainer.global_step} step GB_big: {squared_norm.item()}")


def log_exception(exception, file, place, args):
    # Paths
    log_dir = "./logs/exception/"
    log_file = f"{args.run_id}.log"
    log_file_path = f"{log_dir}{log_file}"

    # Create dir
    os.makedirs(log_dir, exist_ok=True)

    with open(log_file_path, "a") as log_file:
        log_file.write(
            f"*************************************************************************\n"
        )
        log_file.write(f"{exception}\n\n")
        log_file.write(f"Seed:\t\t{args.seed}\n")
        log_file.write(f"File:\t\t{file}\n")
        log_file.write(f"Section:\t{place}\n")
        log_file.write(f"Arguments:\n{args}\n")
        log_file.write(
            f"*************************************************************************\n\n\n\n\n"
        )


def main(args):
    filename = "main.py"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_id = args.run_id
    project_name = args.project_name

    try:
        print(args.train_dataset_filepath)
        train_dataset = load_from_disk(args.train_dataset_filepath)
        val_dataset = load_from_disk(args.val_dataset_filepath)
        test_dataset = load_from_disk(args.test_dataset_filepath)

        test_batch_size = (
            args.test_batch_size if args.test_batch_size else args.batch_size * 2
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=test_batch_size, shuffle=False
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=test_batch_size if "graph" not in args.task else 1, shuffle=False
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        # raise Exception("This is a manually triggered exception.")
    except Exception as e:
        place = "train_dataset, val_dataset, test_dataset creation"
        log_exception(e, filename, place, args)
        raise

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
        max_batch_size=test_batch_size,
        multiple_of=2,
        task=args.task,
    )

    num_batches_per_epoch = len(train_dataset) // args.batch_size
    if len(train_dataset) % args.batch_size != 0:
        num_batches_per_epoch += 1

    total_steps = args.max_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    val_check_interval = max(1, int(total_steps * args.validation_ratio))

    model = Transformer(transformerpp_config)

    model = GPTTrainingModel(
        model,
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
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [ValidateEndCallback(val_dataloader), lr_monitor]

    wandb_logger = WandbLogger(
        project=project_name,
        name=f"{args.task_length}bits-{args.method}-dim{args.n_embd}-lr{args.lr}-batch{args.batch_size}-loss{args.loss_threshold:.3e}-seed{args.seed}",
        id=run_id,
    )
    trainer = Trainer(
        max_steps=args.max_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        accelerator="gpu",
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(args)

    # Try and catch error:
    try:
        trainer.fit(model, train_dataloader, val_dataloader)
    except Exception as e:
        place = "trainer.fit (model, train_dataloader, val_dataloader)"
        log_exception(e, filename, place, args)
        raise

    total_compute = get_compute_from_steps(
        args.n_embd,
        block_size,
        args.task_length,
        trainer.global_step,
        args.batch_size * trainer.world_size,
        n_layers=args.n_layer,
    )

    # Log to wandb
    trainer.logger.experiment.log({"total_compute": total_compute})

    trainer.test(model, test_dataloader)

    wandb.finish()

    print(f"{trainer.global_rank} Finished wandb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dataset_filepath",
        default="./datasets/addition/10/cot_c1/4000000/10162/newline/train_dataset/",
        type=str,
        help="Path to dataset",
    )
    parser.add_argument(
        "--val_dataset_filepath",
        # default=None,
        default="./datasets/addition/10/cot_c1/4000000/10162/newline/train_dataset/",
        type=str,
        help="Path to validation dataset",
    )
    parser.add_argument(
        "--test_dataset_filepath",
        # default=None,
        default="./datasets/addition/10/cot_c1/4000000/10162/newline/test_dataset/",
        type=str,
        help="Path to test dataset (if any)",
    )
    parser.add_argument("--task_length", default=10, type=int, help="Bit length")
    parser.add_argument("--task", default="addition", type=str, help="Method type")
    parser.add_argument("--method", default="cot", type=str, help="Method type")
    parser.add_argument("--seed", default=42, type=int, help="Seed value")
    parser.add_argument("--batch_size", default=800, type=int, help="Batch size")
    parser.add_argument("--test_batch_size", type=int, help="Batch size")
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_head", default=6, type=int)
    parser.add_argument("--n_embd", default=24, type=int)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--max_steps", default=145, type=int)
    parser.add_argument("--compute_budget", default=int(1e13), type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--betas", default=(0.9, 0.95), type=tuple)
    parser.add_argument("--log_gradients", action="store_true", default=False)
    parser.add_argument("--validation_ratio", default=0.1, type=float)
    parser.add_argument(
        "--loss_threshold",
        type=float,
        required=True,
        help="Train Loss Early Stopping Threshold",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        default=False,
        help="Flag for early stopiing",
    )
    parser.add_argument(
        "--threshold_testing",
        action="store_true",
        default=False,
        help="Flag for threshold testing",
    )
    parser.add_argument(
        "--mask_idx", default=2 * 10 + 1, type=int
    )  # You can replace 10 with task_length default value if you want
    parser.add_argument("--precision", default="32", type=str)

    parser.add_argument(
        "--project_name", type=str, required=True, help="WandB Project Name"
    )
    parser.add_argument("--run_id", type=str, required=True, help="WandB Run ID")

    args = parser.parse_args()

    main(args)
