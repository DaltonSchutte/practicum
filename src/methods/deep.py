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


#############
# FUNCTIONS #
#############

def collator(input):
    Xs, ys = zip(*input)
    masks = torch.vstack([
        torch.cat([torch.ones((X.shape[0])),torch.zeros((512-X.shape[0]))]) for X in Xs
    ])
    Xs = torch.dstack([
        torch.cat([X,torch.zeros((512-X.shape[0],X.shape[1]))]) for X in Xs
    ])
    
    return (
        Xs.permute(2,1,0).to(dtype=torch.float32),  # [bsz, n_channels, seqlen]
        masks.to(dtype=torch.long),                 # [bsz, seqlen]
        torch.hstack(ys)                            # [bsz]
    )


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

 
class DLDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long)
        )


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

    def forward(self, x, hc: Optional=None, mask: Optional=None, infer: Optional[bool]=False) -> tuple:
        # This is needed due to a bug in momentfm that causes
        # the model to return None if loaded using classification
        # as the task
        if self.model_type == 'transformer':
            bsz = 1 if infer else self.bsz

            if mask is None:
                mask = torch.ones((bsz, 512)).to(self.device)
            
            x = self.model.normalizer(x, mask=mask)
            x = torch.nan_to_num(x, nan=0, posinf=0, neginf=0)
            x = self.model.tokenizer(x)
            enc = self.model.patch_embedding(x, mask=mask)
            n = enc.shape[2]
            enc = enc.reshape(
                (bsz*self.in_dim, n, self.model.config.d_model)
            )
            out = self.model.encoder(inputs_embeds=enc).last_hidden_state
            out = out.reshape(
                (-1, self.in_dim, n, self.model.config.d_model)
            )
            out = out.permute(0,2,3,1).reshape(
                bsz, n, self.model.config.d_model*self.in_dim
            )
            out = torch.mean(out, dim=1)
            out = self.dropout(out)
            output = self.linear(out)

        elif self.model_type == 'lstm':
            output, hc = self.model(x, hc)
        return output, hc

    def predict(self, x, hc, mask, infer=True):
        x, hc1 = self.forward(x, hc, mask, infer)
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
                "task_name": "classification",
                "n_channels": self.in_dim,
                "freeze_encoder": False,
                "freeze_embedder": True,
                "num_class": 2
            },
        )
        self.linear = nn.Linear(self.in_dim*self.model.config.d_model, 2)
        self.dropout = nn.Dropout(0.1)

    def _prepare_lstm(self):
        self.model = nn.Sequential(
            nn.LSTM(
                self.in_dim,
                self.hid_dim,
                self.n_layers
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
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
        lr: Optional[float]=1e-3,
        eta_min: Optional[float]=1e-6,
        weight_decay: Optional[float]=0.0,
        use_pbar: Optional[bool]=True,
        debug: Optional[bool]=False
    ):
        self.best_loss = 1e9
        self.train_losses = []
        self._f1 = 'Training...'
        self.since_improvement = 0

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay 
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs*len(training_data),
            eta_min=eta_min
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weight.float()
        )

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
                hc,
                debug
            )
            scheduler.step()
            hc, f1, clf_report = self._eval_step(
                valid_data,
                criterion,
                hc,
                debug
            )

            if self.since_improvement == 3:
                break

        print(clf_report)


    def _train_step(
        self,
        training_data,
        optimizer,
        criterion,
        hc=None,
        debug=False
    ):
        self.model.train()
        criterion.to(self.device)
        for batch in tqdm(training_data, total=len(training_data)):
            inputs, masks, labels = batch
            inputs = inputs.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            logits, hc = self.forward(inputs, hc, masks)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            self.train_losses.append(
                loss.cpu().detach().item()
            )

            if self.use_pbar:
                self.pbar.set_postfix(
                    {
                        'Loss': np.mean(self.train_losses[-100:]),
                        'F1': self._f1
                    }
                )
            if debug:
                break

    def _eval_step(
        self,
        valid_data,
        criterion,
        hc,
        debug
    ):
        self.model.eval()
        epoch_preds = []
        epoch_labels = []
        epoch_logits = []
        criterion.to('cpu')
        with torch.no_grad():
            for batch in tqdm(valid_data, total=len(valid_data)):
                inputs, masks, labels = batch
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                logits, hc = self.forward(inputs, hc, masks, infer=True)

                epoch_preds.append(logits.softmax(1).argmax(1).cpu().numpy())
                epoch_labels.append(labels.cpu().numpy())
                epoch_logits.append(logits.cpu().reshape(-1,2).numpy())

                if debug:
                    break

            epoch_preds = np.concatenate(epoch_preds, axis=0)
            epoch_labels = np.concatenate(epoch_labels, axis=0)
            epoch_logits = np.concatenate(epoch_logits, axis=0)

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
            print(clf_report)

            loss = criterion(
                torch.tensor(epoch_logits),
                torch.tensor(epoch_labels)
            ).cpu().detach().item()

            if loss < self.best_loss:
                torch.save(
                    self.state_dict(),
                    self.save_dir
                )
                self.best_loss = loss
                self.since_improvement = 0
            else:
                self.since_improvement += 1
        
        self._f1 = epoch_f1

        if self.use_pbar:
            self.pbar.set_postfix(
                {
                    'Loss': loss,
                    'F1': self._f1
                }
            )
            self.pbar.update(1)
        return hc, epoch_f1, clf_report
