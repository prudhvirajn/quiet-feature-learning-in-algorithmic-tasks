from typing import Dict

import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from utils import WarmupCosineAnnealingLR, ConstantLR, WarmupConstantLR

# from mup import MuAdamW

import pytorch_lightning as pl
import logging


class GPTTrainingModel(pl.LightningModule):
    def __init__(
        self,
        model,
        warmup_steps,
        total_steps,
        max_lr,
        task_length,
        betas,
        weight_decay,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        repetitions: int = 1,
        method="normal",
        log_gradients=False,
        log_activations=False,
    ):
        super(GPTTrainingModel, self).__init__()

        self.model = model
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.task_length = task_length
        self.repetitions = repetitions
        self.method = method
        self.betas = betas
        self.weight_decay = weight_decay
        self.log_gradients = log_gradients
        self.log_activations = log_activations

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.local_logger = logging.getLogger(__name__)

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader

    def test_dataloader(self):
        return self.test_dataloader

    def training_step(self, batch, batch_idx):
        current_step = self.trainer.global_step
        input_ids = torch.vstack(batch["ids"]).T.to(self.device)
        loss_mask = torch.vstack(batch["attention_mask"]).T.to(self.device)

        # logits, loss = self.model(
        #     input_ids,
        #     targets=input_ids[:, 1:].long(),
        #     loss_mask=loss_mask[:, 1:],
        # )
        logits, loss, _ = self.model(
            input_ids,
            targets=input_ids[:, 1:].long(),
            loss_mask=loss_mask[:, 1:],
        )
        
        # Infer batch size from input_ids
        batch_size = input_ids.shape[0]

        # Logging loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        current_step = self.trainer.global_step
        input_ids = torch.vstack(batch["ids"]).T.to(self.device)
        loss_mask = torch.vstack(batch["attention_mask"]).T.to(self.device)

        # Infer batch size from input_ids
        batch_size = input_ids.shape[0]

        if batch_idx == 0:
            logits, loss, _ = self.model(
                input_ids,
                targets=input_ids[:, 1:].long(),
                loss_mask=loss_mask[:, 1:],
                capture_scores=self.log_activations
            )
        else:
            logits, loss, _ = self.model(
                input_ids,
                targets=input_ids[:, 1:].long(),
                loss_mask=loss_mask[:, 1:],
                capture_scores=False
            )

        # Logging loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = torch.vstack(batch["ids"]).T.to(self.device)
        attention_mask = torch.vstack(batch["attention_mask"]).T.to(self.device)
        
        total_tokens = self.model.params.max_seq_len // self.repetitions # self.model.config.block_size 
        
        target_token = self.model.params.vocab_size - 2
        num_input_tokens = (input_ids == target_token).nonzero(as_tuple=True)[1][0]

        # if "addition" in self.model.params.task:
        #     num_input_tokens = 2 * self.task_length + 2
        #     result_tokens = self.task_length + 2
        # elif "multiplication" in self.model.params.task:
        #     num_input_tokens = 2 * self.task_length + 2
        #     result_tokens = 2 * self.task_length + 1
        # elif "parity" == self.model.params.task:
        #     num_input_tokens = 2 * self.task_length
        #     result_tokens = 2
        # elif "sorting" == self.model.params.task:
        #     num_input_tokens = self.task_length + 1
        #     result_tokens = self.task_length + 1
        # elif "graph" in self.model.params.task:
        #     target_token = self.model.params.vocab_size - 2
        #     num_input_tokens = (input_ids == target_token).nonzero(as_tuple=True)[1][0]
        #     result_tokens = total_tokens - num_input_tokens
        # elif "graph_longest_path" == self.model.params.task:
        #     target_token = self.model.params.vocab_size - 2
        #     num_input_tokens = (input_ids == target_token).nonzero(as_tuple=True)[1][0]
        #     result_tokens = total_tokens - num_input_tokens

        remaining_tokens = total_tokens - num_input_tokens
        input_tokens = input_ids[:, :num_input_tokens]
        # label_ids = input_ids[attention_mask == 1].reshape(input_ids.shape[0], -1)

        # outputs = self.model.generate(
        #     input_tokens, remaining_tokens, temperature=1e-9, top_k=1
        # )[attention_mask == 1].reshape(input_ids.shape[0], -1)

        label_ids = input_ids * attention_mask
        
        outputs = self.model.generate(
            input_tokens, remaining_tokens, temperature=1e-9, top_k=1
        ) * attention_mask
        
        check = torch.all(
            outputs == label_ids, 1
        )
        test_accuracy_result = torch.sum(check).item() / input_ids.shape[0]

        metrics = {"test_acc_result": test_accuracy_result}

        # if "cot" in self.method:
        #     reasoning_label_ids = input_ids[
        #         :, num_input_tokens : total_tokens - result_tokens
        #     ]
        #     reasoning_check = torch.all(
        #         outputs[:, num_input_tokens : total_tokens - result_tokens]
        #         == reasoning_label_ids,
        #         1,
        #     )
        #     test_accuracy_reasoning = (
        #         torch.sum(reasoning_check).item() / input_ids.shape[0]
        #     )
        #     metrics["test_acc_reasoning"] = test_accuracy_reasoning

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # optimizer = MuAdamW(
        #     self.parameters(),
        #     lr=self.max_lr,
        #     betas=self.betas,
        #     weight_decay=self.weight_decay,
        # )

        optimizer = AdamW(
            self.parameters(),
            lr=self.max_lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        scheduler = {
            "scheduler": WarmupCosineAnnealingLR(
                optimizer,
                warmup_steps=self.warmup_steps,
                total_steps=self.total_steps,
                eta_min=0.1 * self.max_lr,
            ),
            "interval": "step",  # Step after every batch/iteration
        }
        return [optimizer], [scheduler]
