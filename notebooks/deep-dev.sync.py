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
import pickle
import random
import datetime
from string import Template
from collections import namedtuple

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from momentfm import MOMENTPipeline

import sys
sys.path.insert(0, '..')
from src.data import TimeSeries
from src.methods.deep import (
    DeepStoppingModel,
    TimeSeriesDataset,
    DLDataset,
    collator
)
from src.eval import (
    mean_time_from_event,
    classification_metrics
)

# %%
SEED = 3141
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 8

# %%
train_ts = TimeSeries.from_csv(
    'pandas',
    '../data/blood-refrigerator/train.csv'
)
valid_ts = TimeSeries.from_csv(
    'pandas',
    '../data/blood-refrigerator/val.csv'
)
test_ts = TimeSeries.from_csv(
    'pandas',
    '../data/blood-refrigerator/test.csv'
)

train_ts.parse_datetime('timestamp')
valid_ts.parse_datetime('timestamp')
test_ts.parse_datetime('timestamp')

train_ts.split_by_day()
valid_ts.split_by_day()
test_ts.split_by_day()


# %%
temp = pd.concat(
    train_ts.time_series[k].drop(
        columns=['timestamp','PW_0.5h','date','time']
    ) for k in train_ts.time_series.keys()
)

FEATURE_COLS = [
    c for c in temp.columns if np.std(temp[c])!=0
]
LABEL_COL = 'PW_0.5h'

temp = None
del temp
len(FEATURE_COLS)

# %%
def get_time_windows(df, window={'minutes': 20}):
    tf = '%Y-%m-%d %H:%M:%S.%f'
    windows = []
    window_size = datetime.timedelta(**window)
    df.reset_index(drop=True, inplace=True)
    for i, row in df.iterrows():
        start_t = datetime.datetime.strptime(row['timestamp'], tf)
        for j, row2 in df.iloc[i:].iterrows():
            end_t = datetime.datetime.strptime(row2['timestamp'], tf)
            if end_t - start_t >= window_size:
                try:
                    # label = df['PW_0.5h'].iloc[j]
                    label = df['PW_0.5h'].iloc[j]
                except IndexError:
                    label = df['PW_0.5h'].iloc[-1]
                except err:
                    print(err)
                windows.append(
                    (df.iloc[i:j+1], label)
                )
                break
    return windows

# %%
save_dir = '../data/blood-refrigerator/preprocessed'
if not os.path.isfile(os.path.join(save_dir, 'X_train.pkl')):
    print('Making datasets...')
    train_windows = {
        dt: get_time_windows(ts) for dt, ts in train_ts.time_series.items()
    }

    valid_windows = {
        dt: get_time_windows(ts) for dt, ts in valid_ts.time_series.items()
    }
    test_windows = {
        dt: get_time_windows(ts) for dt, ts in test_ts.time_series.items()
    }

    X_train = [
        df[FEATURE_COLS].values for pr in train_windows.values() for (df,_) in pr
    ]
    y_train = [
        l for pr in train_windows.values() for (_,l) in pr
    ]
    X_valid = [
        df[FEATURE_COLS].values for pr in valid_windows.values() for (df,_) in pr
    ]
    y_valid = [
        l for pr in valid_windows.values() for (_,l) in pr
    ]
    X_test = [
        df[FEATURE_COLS].values for pr in test_windows.values() for (df,_) in pr
    ]
    y_test = [
        l for pr in test_windows.values() for (_,l) in pr
    ]

    pickle.dump(X_train, open(os.path.join(save_dir, 'X_train.pkl'), 'wb'))
    pickle.dump(y_train, open(os.path.join(save_dir, 'y_train.pkl'), 'wb'))
    pickle.dump(X_valid, open(os.path.join(save_dir, 'X_valid.pkl'), 'wb'))
    pickle.dump(y_valid, open(os.path.join(save_dir, 'y_valid.pkl'), 'wb'))
    pickle.dump(X_test, open(os.path.join(save_dir, 'X_test.pkl'), 'wb'))
    pickle.dump(y_test, open(os.path.join(save_dir, 'y_test.pkl'), 'wb'))
else:
    print('Loading...')
    X_train = pickle.load(open(os.path.join(save_dir, 'X_train.pkl'), 'rb'))
    y_train = pickle.load(open(os.path.join(save_dir, 'y_train.pkl'), 'rb'))
    X_valid = pickle.load(open(os.path.join(save_dir, 'X_valid.pkl'), 'rb'))
    y_valid = pickle.load(open(os.path.join(save_dir, 'y_valid.pkl'), 'rb'))
    X_test = pickle.load(open(os.path.join(save_dir, 'X_test.pkl'), 'rb'))
    y_test = pickle.load(open(os.path.join(save_dir, 'y_test.pkl'), 'rb'))


{
    'train': (len(X_train), len(y_train), np.mean(y_train)),
    'valid': (len(X_valid), len(y_valid), np.mean(y_valid)),
    'test': (len(X_test), len(y_test), np.mean(y_test))
}


# %%
train_ds = DLDataset(X_train, y_train)
valid_ds = DLDataset(X_valid, y_valid)
test_ds = DLDataset(X_test, y_test)

train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
    drop_last=True
)
valid_dl = DataLoader(
    valid_ds,
    collate_fn=collator
)
test_dl = DataLoader(
    test_ds,
    collate_fn=collator
)

# %%
for x, mask, y in train_dl:
    break

x.shape, mask.shape, y.shape, x.dtype

# %%
model = DeepStoppingModel(
    'transformer',
    len(FEATURE_COLS),
    device='cuda',
    save_dir='../results/deep/transformer.pt',
    bsz=BATCH_SIZE
)

# %%
pw1 = np.mean(train_ds.y)**2
pw0 = 1-pw1
pw0,pw1
weights = torch.tensor([1/pw0,1/pw1])
weights /= weights.sum()
weights

# %%
# Expects tensors in [bsz, in_channels, seqlen]
model.train(
    5,
    train_dl,
    valid_dl,
    class_weight=weights,
    lr=1e-3,
    eta_min=1e-6,
    weight_decay=0.1
)
