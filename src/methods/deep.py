"""
Deep learning methods
"""
import os
from typing import Optional

import torch
from torch import nn, optim
from torch.nn import functional as F

from momentfm import MOMENTPipeline


###########
# GLOBALS #
###########

MOMENT = "AutonLab/MOMENT-1-large"


###########
# CLASSES #
###########

class DeepStoppingModel(nn.Module):
    def __init__(
        self,
        model_type: str,
        in_dim: int,
        hid_dim: Optional[int]=64,
        n_layers: Optional[int]=1,
        bsz: Optional[int]=1
    ):
        super().__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bsz = bsz

        if model_type == 'transformer':
            self._prepare_transformer()
        elif model_type == 'lstm':
            self._prepare_lstm()
        else:
            raise ValueError(
                f"Received model={model}, expected lstm or transformer"
            )

    def forward(self, x, hc: Optional=None) -> tuple:
        if self.model_type == 'transformer':
            output = self.model(x).logits
        elif self.model_type == 'lstm':
            output, hc = self.model(x, hc)
        return output, hc

    def _prepare_transformer(self):
        self.model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.in_dim,
                "num_class": 2
            },
        )

    def _prepare_lstm(self):
        self.model = nn.Sequential(
            nn.LSTM(
                self.in_dim,
                self.hid_dim,
                self.n_layers
            ),
            nn.ReLU(),
            nn.Linear(hid_dim, 2),
            nn.Softmax()
        )

    def init(self) -> tuple:
        if self.model_type == 'transformer':
            self.model.init()
            return (None, None)
        elif self.model_type == 'lstm':
            return (
                torch.randn(
                    self.n_layers,
                    self.bsz,
                    self.hid_dim
                ),
                torch.randn(
                    self.n_layers,
                    self.bsz,
                    self.hid_dim
                )
            )

    def train(
        self,
        epochs: int,
        lr: float,
        wd: float,
        training_data,
        valid_data
    ):
        raise NotImplementedError
