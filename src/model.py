from collections import defaultdict

import numpy as np
import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanSquaredError, Metric

from .cpm import cpm


class CorrMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[])
        self.add_state("target", default=[])

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds.append(preds.detach())
        self.target.append(target.detach())

    def compute(self):
        if len(self.preds) == 0:
            return torch.tensor(0)
        preds = torch.cat(self.preds)
        target = torch.cat(self.target)
        assert preds.shape == target.shape

        vx = preds - torch.mean(preds)
        vy = target - torch.mean(target)
        corr = torch.sum(vx * vy) / (1e-7 + torch.norm(vx) * torch.norm(vy))

        return corr


class RegressionModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float = 0.01,
        beta: float = 0.0,
        node_entropy: dict[str, np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta = beta

        for key, value in node_entropy.items():
            self.register_buffer("entropy_" + key, torch.from_numpy(value))

        self.criterion = nn.MSELoss()
        self.train_metric = CorrMetric()
        self.val_metric = CorrMetric()
        self.mse = MeanSquaredError()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        def lr_scaler(epoch):
            warmup_epoch = 100
            if epoch < warmup_epoch:
                # warm up lr
                lr_scale = epoch / warmup_epoch
            else:
                lr_scale = 1.0

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scaler)

        return [optimizer], [scheduler]

    def forward(self, batch):
        x = batch["x"]
        meta = self.get_meta_from_batch(batch)
        task_key = meta["task_name"]

        outputs = self.model(x)

        y_hat = outputs["preds"]
        last = outputs["last"]
        mask = outputs.get("mask", None)
        mask_logits = outputs.get("mask_logits", None)
        assert y_hat.dim() == 1

        y = batch["y"]
        assert y.dim() == 1

        max_term = -self.criterion(y_hat, y)

        density = mask.mean()
        node_entropy = getattr(self, "entropy_" + task_key)
        min_term = (torch.sigmoid(mask_logits.squeeze()) * node_entropy).mean()

        loss = -max_term + self.beta * min_term

        return (loss, max_term, min_term), (y_hat, y, last), density

    def training_step(self, batch, batch_idx):
        (loss, max_term, min_term), (y_hat, y, _), density = self.forward(batch)
        self.train_metric(y_hat, y)
        self.log("train/loss", loss)
        self.log("train/density", density)
        self.log("train/max_term", max_term)
        self.log("train/min_term", min_term)

        return loss

    def on_train_epoch_end(self):
        self.log("train/corr", self.train_metric)

    def validation_step(self, batch, batch_idx):
        (loss, max_term, min_term), (y_hat, y, _), density = self.forward(batch)

        meta = self.get_meta_from_batch(batch)
        self.val_metric(y_hat, y)
        self.log("val/loss", loss)
        self.log("val/density", density)
        self.log("val/max_term", max_term)
        self.log("val/min_term", min_term)

        self.mse(y_hat, y)
        self.log("val/mse", self.mse)

    def on_validation_epoch_end(self):
        self.log("val/corr", self.val_metric)

    def on_test_start(self):
        self.weights = defaultdict(list)
        self.labels = defaultdict(list)

    def get_meta_from_batch(self, batch):
        meta = batch["meta"]
        if isinstance(meta, list):
            meta = meta[0]
        elif isinstance(meta, dict):
            meta = {k: v[0] for k, v in meta.items()}
        return meta

    def test_step(self, batch, batch_idx):
        x = batch["x"]
        bs = x.size(0)
        seq_len = x.size(-1)

        _, (y_hat, y, weights), _ = self(batch)

        meta = self.get_meta_from_batch(batch)
        task: str = meta["task_name"]
        dataset_name: str = meta["dataset_name"]
        self.weights[dataset_name + "-" + task].append(weights.cpu().numpy())
        self.labels[dataset_name + "-" + task].append(batch["y"].cpu().numpy())

    def on_test_end(self) -> None:
        for key in self.weights.keys():
            weights = self.weights[key]
            labels = self.labels[key]
            weights = np.concatenate(weights)
            weights = (weights - weights.mean()) / (1e-7 + weights.std())
            labels = np.concatenate(labels)
            weights = np.transpose(weights)
            X = np.concatenate([weights, -weights], axis=0)
            Y = labels[..., None]
            dataset, task = key.split("-")

            train_ids, test_ids = self.trainer.datamodule.get_train_test_split(
                dataset, task
            )
            train_test_split = {
                "train_id": np.asarray(train_ids),
                "test_id": np.asarray(test_ids),
            }
            R, P = cpm(X, Y, train_test_split)
            print("CPM_R:", R.item())
