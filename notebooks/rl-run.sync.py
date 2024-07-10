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
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import torch

import sys
sys.path.insert(0, '..')
from src.data import TimeSeries
from src.methods import rl
from src.environment import TimeSeriesEnv

# %%
# GLOBALS
SEED = 3141
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

warnings.filterwarnings('ignore', category=FutureWarning)

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

    # Make environments
    train_env = TimeSeriesEnv(train_ts, FEATURE_COLS, LABEL_COL)
    valid_env = TimeSeriesEnv(valid_ts, FEATURE_COLS, LABEL_COL)
    test_env = TimeSeriesEnv(test_ts, FEATURE_COLS, LABEL_COL)

    # Model prep
    hid_dim = 0
    for i in range(10):
        if 2**i > len(FEATURE_COLS):
            hid_dim = 2**i
            break
    print('Features: ', len(FEATURE_COLS))
    print('Hidden Dim: ',hid_dim)
    alpha_stop = rl.NeuralNetGuidedMCTS(
        in_dim=len(FEATURE_COLS),
        hid_dim=hid_dim,
        save_dir=os.path.join('../results/rl',dir),
        n_actions=2,
        n_sim=100,
        lr=1e-4,
        weight_decay=0.01,
        gamma=0.99,
        bsz=128,
        device='cpu'
    )

    # Train
    epochs = 30
    train_actions, train_rewards = alpha_stop.train(epochs, train_env, valid_env)

    # Load and test best model
    alpha_stop.net.load_state_dict(
        torch.load(
            os.path.join('../results/rl',dir,'network.pt')
        )
    )
    alpha_stop.mcts = pickle.load(open(os.path.join('../results/rl',dir,'mcts.pkl'),'rb'))
    test_actions, test_rewards = alpha_stop.run(test_env)
    
    # Save output
    pickle.dump(train_actions, open(os.path.join('../results/rl',dir,'train_actions.pkl'), 'wb'))
    pickle.dump(train_rewards, open(os.path.join('../results/rl',dir,'train_rewards.pkl'), 'wb'))
    pickle.dump(test_actions, open(os.path.join('../results/rl',dir,'test_actions.pkl'), 'wb'))
    pickle.dump(test_rewards, open(os.path.join('../results/rl',dir,'test_rewards.pkl'), 'wb'))

print("<< PROCESS COMPELTE >>")
