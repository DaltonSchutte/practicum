# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
from string import Template
from collections import namedtuple

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '..')
from src.data import TimeSeries
from src.methods.deep import (
    DeepStoppingModel,
    TimeSeriesDataset
)
from src.eval import (
    mean_time_from_event,
    classification_metrics
)

# %%

br_train = TimeSeries.from_csv(
    'pandas',
    '../data/blood-refrigerator/train.csv'
)
br_dev = TimeSeries.from_csv(
    'pandas',
    '../data/blood-refrigerator/val.csv'
)
br_test = TimeSeries.from_csv(
    'pandas',
    '../data/blood-refrigerator/test.csv'
)
print(br_train.shape, br_dev.shape, br_test.shape)
br_dev.head()

# %%
br_train.parse_datetime('timestamp')
br_dev.parse_datetime('timestamp')
br_test.parse_datetime('timestamp')

br_train.split_by_day()
br_dev.split_by_day()
br_test.split_by_day()

len(br_train.time_series), len(br_dev.time_series), len(br_test.time_series)

# %%
splits = [0.25, 0.5, 0.75, 1.0]
br_train_splits = {}

for pct in splits:
    n_days = len(br_train.time_series)
    train_days = list(br_train.time_series.keys())[-int(pct*n_days):]
    y = pd.concat([
        br_train.time_series[k]['PW_0.5h'] for k in train_days
    ])
    X = pd.concat([
        br_train.time_series[k].drop(
            columns=['timestamp','PW_0.5h','date','time']
        ) for k in train_days
    ])

    # Drop std=0 variables
    X = X[
        [c for c in X.columns if np.std(X[c]) != 0]
    ]

    if pct == 0.25:
        keep_cols = X.columns
        
    br_train_splits.update(
        {
            str(pct):
            {
                'X': X.values,
                'y': y.values,
                'cols': keep_cols.tolist()
            }
        }
    )
    print(f"{pct}\t-\t{X.shape}\t-\t{y.shape}\t-\n{keep_cols.tolist()}\n")

# %%
br_dev_data = {
    'X': {dt: x[keep_cols].values for dt, x in br_dev.time_series.items()},
    'y': {dt: x['PW_0.5h'].values for dt, x in br_dev.time_series.items()}
}
br_test_data = {
    'X': {dt: x[keep_cols].values for dt, x in br_test.time_series.items()},
    'y': {dt: x['PW_0.5h'].values for dt, x in br_test.time_series.items()}
}

# %%
br_train_X = []
br_train_y = []

br_025 = br_train_splits['0.25']

print(br_025['X'].shape)

br_train_X = np.lib.stride_tricks.sliding_window_view(
    br_025['X'],
    (512, 12)
)
br_train_y = np.lib.stride_tricks.sliding_window_view(
    br_025['y'],
    (512)
)
    
br_train_X = torch.tensor(br_train_X).squeeze().reshape(-1,12,512).type(torch.float32).requires_grad_()
br_train_y = torch.tensor(br_train_y).squeeze().type(torch.long)
print(br_train_X.type(), br_train_y.type())
br_train_X.shape, br_train_y.shape

# %%
br_val_X = torch.tensor(np.lib.stride_tricks.sliding_window_view(
    np.concatenate(
        list(br_dev_data['X'].values())
    ),
    (512,12)
)).squeeze().reshape(-1,12,512).type(torch.float32).requires_grad_()
br_val_y = torch.tensor(np.lib.stride_tricks.sliding_window_view(
    np.concatenate(
        list(br_dev_data['y'].values())
    ),
    (512)
)).type(torch.long)
br_test_X = torch.tensor(np.lib.stride_tricks.sliding_window_view(
    np.concatenate(
        list(br_test_data['X'].values())
    ),
    (512,12)
)).squeeze().reshape(-1,12,512).type(torch.float32).requires_grad_()
br_test_y = torch.tensor(np.lib.stride_tricks.sliding_window_view(
    np.concatenate(
        list(br_test_data['y'].values())
    ),
    (512)
)).type(torch.long)

print(br_val_X.type(), br_val_y.type(), br_test_X.type(), br_test_y.type())
br_val_X.shape, br_val_y.shape, br_test_X.shape, br_test_y.shape
        
# %%
br025_ds = TimeSeriesDataset(br_train_X, br_train_y)
train_dl = DataLoader(
    br025_ds,
    batch_size=8,
    shuffle=True
)

# %%
br_dev_ds = TimeSeriesDataset(br_val_X, br_val_y)
br_test_ds = TimeSeriesDataset(br_test_X, br_test_y)

valid_dl = DataLoader(
    br_dev_ds,
    batch_size=1,
    shuffle=False
)
test_dl = DataLoader(
    br_test_ds,
    batch_size=1,
    shuffle=False
)

# %%
model = DeepStoppingModel(
    'transformer',
    12,
    device='cuda'
)

# %%
# Expects tensors in [bsz, in_channels, seqlen]
model.train(
    5,
    train_dl,
    valid_dl,
    lr=0.0001,
    momentum=0.9,
    dampening=0.0,
    weight_decay=0.01,
    nesterov=False
)
