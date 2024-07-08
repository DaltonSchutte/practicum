"""
Deep learning methods
"""
import os
from typing import Optional

from tqdm.auto import tqdm

from sklearn.metrics import f1_score, classification_report

import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset

from momentfm import MOMENTPipeline


###########
# GLOBALS #
###########

MOMENT = "AutonLab/MOMENT-1-large"


###########
# CLASSES #
###########


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y
        
    def __len__(self):
        return self.y.shape[0]
 

class DeepStoppingModel(nn.Module):
    def __init__(
        self,
        model_type: str,
        in_dim: int,
        hid_dim: Optional[int]=64,
        n_layers: Optional[int]=1,
        bsz: Optional[int]=1,
        save_dir: Optional[str]='./deep-model.pt',
        device: Optional[str]='cpu'
    ):
        super().__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bsz = bsz
        self.save_dir = save_dir
        self.device = device

        if model_type == 'transformer':
            self._prepare_transformer()
        elif model_type == 'lstm':
            self._prepare_lstm()
        else:
            raise ValueError(
                f"Received model={model}, expected lstm or transformer"
            )
        self.to(device)

    def forward(self, x, hc: Optional=None) -> tuple:
        if self.model_type == 'transformer':
            x = self.model(x).reconstruction.reshape(-1,512,self.in_dim)
            output = self.linear(x).reshape(-1,2,512)
        elif self.model_type == 'lstm':
            output, hc = self.model(x, hc)
        return output, hc

    def predict(self, x, hc):
        x, hc1 = self.forward(x, hc)
        preds = x.softmax(1).argmax(1)
        return (preds, hc1)

    def to(self, device):
        if self.model_type == 'transformer':
            self.model.to(self.device)
            self.linear.to(self.device)
        elif self.model_type == 'lstm':
            self.model.to(self.device)

    def _prepare_transformer(self):
        self.model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "reconstruction",
                "n_channels": self.in_dim,
                "freeze_encoder": True,
                "freeze_embedder": True
            },
        )
        self.linear = nn.Linear(self.in_dim, 2)

    def _prepare_lstm(self):
        self.model = nn.Sequential(
            nn.LSTM(
                self.in_dim,
                self.hid_dim,
                self.n_layers
            ),
            nn.ReLU(),
            nn.Linear(hid_dim, 2)
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
        training_data,
        valid_data,
        class_weight: Optional[torch.Tensor]=torch.tensor([1.0,1.0]),
        lr: Optional[float]=0.001,
        momentum: Optional[float]=0.9,
        dampening: Optional[float]=0.0,
        weight_decay: Optional[float]=0.0,
        nesterov: Optional[bool]=True,
        use_pbar: Optional[bool]=True
    ):
        self.best_loss = 1e9
        self.train_losses = []
        self._f1 = 'Training...'

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weight
        ).to(self.device)

        self.use_pbar = use_pbar
        self.pbar = None
        if self.use_pbar:
            self.pbar = tqdm(total=epochs)

        hc = self.model.init()

        for epoch in range(epochs):
            self._train_step(
                training_data,
                optimizer,
                criterion,
                hc
            )
            hc, f1, clf_report = self._eval_step(
                valid_data,
                criterion,
                hc
            )

        print(clf_report)


    def _train_step(
        self,
        training_data,
        optimizer,
        criterion,
        hc,
    ):
        self.model.train()
        for batch in tqdm(training_data, total=int(14001/8)):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            logits, hc = self.forward(inputs, hc)
            loss = criterion(logits, labels)

            self.train_losses.append(
                loss.cpu().detach().item()
            )

            loss.backward()
            optimizer.step()

            if self.use_pbar:
                self.pbar.set_postfix(
                    {
                        'Loss': np.mean(self.train_losses[-100:]),
                        'F1': self._f1
                    }
                )
            break
            

    def _eval_step(
        self,
        valid_data,
        criterion,
        hc
    ):
        self.model.eval()
        epoch_preds = []
        epoch_labels = []
        epoch_logits = []
        with torch.no_grad():
            for batch in tqdm(valid_data, total=13481):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits, hc = self.forward(inputs, hc)

                # Both should be [512] shape
                epoch_preds += logits.softmax(1).argmax(1).squeeze().cpu().numpy().tolist()
                epoch_labels += labels.squeeze().cpu().numpy().tolist()
                epoch_logits += logits.cpu().numpy().tolist()
                break

            print(epoch_labels, epoch_preds)
            epoch_f1 = f1_score(
                epoch_labels,
                epoch_preds,
                zero_division=0.0
            )
            clf_report = classification_report(
                epoch_labels,
                epoch_preds,
                zero_division=0.0
            )

            loss = criterion(
                torch.tensor(epoch_logits),
                torch.tensor(epoch_labels)
            ).cpu().detach().item()

            if loss < self.best_loss:
                torch.save(
                    self.model,
                    save_dir
                )
                self.best_loss = loss
        
        self._f1 = epoch_f1

        if self.use_pbar:
            self.pbar.set_postfix(
                {
                    'Loss': loss,
                    'F1': self._f1
                }
            )
        return hc, epoch_f1, clf_report
