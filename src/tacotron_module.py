from typing import Tuple
from omegaconf import DictConfig

from pytorch_lightning import LightningModule

from model import Tacotron2
from loss_function import Tacotron2Loss
import torch


class TacotronModuel(LightningModule):
    def __init__(self, configs: DictConfig):
        super(TacotronModuel, self).__init__()
        self.configs = configs
        self.model = Tacotron2(configs.model)
        self.criterion = Tacotron2Loss()

    def training_step(self, batch: Tuple, batch_idx: int):
        x, y = self.model.parse_batch(batch)
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def on_train_start(self):
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    def validation_step(self, batch: Tuple, batch_idx: int):
        x, y = self.model.parse_batch(batch)
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self.log('valid_loss', loss)
        return {'valid_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.configs.model.lr,
            weight_decay=self.configs.model.weight_decay
        )
        return [optimizer], []

    def get_lr(self):
        for g in self.optimizers().param_groups:
            return g["lr"]