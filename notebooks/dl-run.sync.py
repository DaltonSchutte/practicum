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
from torch.utils.data import Dataset, DataLoader

from momentfm import MOMENTPipeline

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
SEED = 3141
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 4

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
                    label = df['PW_0.5h'].iloc[j+1]
                except IndexError:
                    label = df['PW_0.5h'].iloc[-1]
                except err:
                    print(err)
                windows.append(
                    (df.iloc[i:j+1], label)
                )
                break
    return windows

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

def collator(input):
    Xs, ys = zip(*input)
    Xs = torch.dstack([
        torch.cat([X,torch.zeros((512-X.shape[0],X.shape[1]))]) for X in Xs
    ])
    return (Xs.permute(2,1,0).to(dtype=torch.float32), torch.hstack(ys))




# %%
for dir in os.listdir('../data'):
    if dir == 'wrapper-machine':
        continue

    print(dir.upper())
    # Data loading
    data_dir_path = os.path.join('../data', dir)
    if dir == 'nitrogen-generator':
        tf = '%Y-%m-%d %H:%M:%S'
    else:
        tf = '%Y-%m-%d %H:%M:%S.%f'

    train_ts = TimeSeries.from_csv(
        'pandas',
        os.path.join(data_dir_path, 'train.csv')
    )
    valid_ts = TimeSeries.from_csv(
        'pandas',
        os.path.join(data_dir_path, 'val.csv')
    )
    test_ts = TimeSeries.from_csv(
        'pandas',
        os.path.join(data_dir_path, 'test.csv')
    )

    # Data prep
    train_ts.parse_datetime('timestamp', tf)
    valid_ts.parse_datetime('timestamp', tf)
    test_ts.parse_datetime('timestamp', tf)

    train_ts.split_by_day()
    valid_ts.split_by_day()
    test_ts.split_by_day()

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

    save_dir = os.path.join('../data',dir,'preprocessed')
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


    print({
        'train': (len(X_train), len(y_train), np.mean(y_train)),
        'valid': (len(X_valid), len(y_valid), np.mean(y_valid)),
        'test': (len(X_test), len(y_test), np.mean(y_test))
    })

    train_ds = DLDataset(X_train, y_train)
    valid_ds = DLDataset(X_valid, y_valid)
    test_ds = DLDataset(X_test, y_test)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator
    )
    valid_dl = DataLoader(
        valid_ds
    )
    test_dl = DataLoader(
        test_ds
    )

    model = DeepStoppingModel(
        'transformer',
        len(FEATURE_COLS),
        device='cuda',
        save_dir=os.path.join('../results',dir,'transformer.pt'),
        bsz=BATCH_SIZE
    )

    # Loss weighting to manage class imbalance
    pw1 = np.mean(train_ds.y)
    pw0 = 1-pw1
    pw0,pw1
    weights = torch.tensor([1/pw0,1/pw1])
    weights /= weights.sum()
    weights

    model.train(
        5,
        train_dl,
        valid_dl,
        class_weight=weights,
        lr=1e-3,
        eta_min=1e-6,
        weight_decay=0.01
    )
