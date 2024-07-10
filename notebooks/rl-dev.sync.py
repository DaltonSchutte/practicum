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
train_env = TimeSeriesEnv(train_ts, FEATURE_COLS, LABEL_COL)
valid_env = TimeSeriesEnv(valid_ts, FEATURE_COLS, LABEL_COL)
test_env = TimeSeriesEnv(test_ts, FEATURE_COLS, LABEL_COL)

# %%
alpha_stop = rl.NeuralNetGuidedMCTS(
    in_dim=len(FEATURE_COLS),
    hid_dim=64,
    save_dir='../results/rl/br',
    n_actions=2,
    n_sim=10,
    lr=1e-3,
    weight_decay=0.1,
    gamma=0.999,
    bsz=32,
    device='cpu'
)

# %%
train_actions, train_rewards = alpha_stop.train(5, train_env, valid_env)

# %%
for (dt, a) in train_actions.items():
    rs = train_rewards[dt]
    print(dt, sum(a), sum(rs))

# %%
test_actions, test_rewards = alpha_stop.run(test_env)

# %%
pickle.dump(train_actions, open(os.path.join('../results/rl/br','train_actions.pkl'), 'wb'))
pickle.dump(train_rewards, open(os.path.join('../results/rl/br','train_rewards.pkl'), 'wb'))
pickle.dump(test_actions, open(os.path.join('../results/rl/br','test_actions.pkl'), 'wb'))
pickle.dump(test_rewards, open(os.path.join('../results/rl/br','test_rewards.pkl'), 'wb'))

# %%
sns.lineplot(
    {dt: sum(r) for dt, r in test_rewards.items()}
)
