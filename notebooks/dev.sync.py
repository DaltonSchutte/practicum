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

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '../')
from src.data import (
    TimeSeries,
    parse_datetime
)
from src.methods import spc


# %%
SEED = 3141

# %%
br = {
    'train': TimeSeries.from_csv(
        'pandas',
        '../data/blood-refrigerator/train.csv',
        sep=',',
    ),
    'test': TimeSeries.from_csv(
        'pandas',
        '../data/blood-refrigerator/test.csv',
        sep=',',
    )
}

# %%
for t,d in br.items():
    print(t, d.shape)

# %%
for t,d in br.items():
    d.parse_datetime('timestamp')

# %%
days = 0
for t,d in br.items():
    d.split_by_day()
    days += len(d.time_series)
days

# %%
counts = []
totals = []
for t, df in br.items():
    for date, d in df.time_series.items():
        counts.append(len(d[d['PW_0.5h']==1]))
        totals.append(len(d))
counts[:10], totals[:10]

# %%
id2day = {i:d for i,d in enumerate(br['train'].time_series.keys())}
train_days = [id2day[i] for i,c in enumerate(counts[:len(id2day)]) if c == 0]

y = pd.concat([
    br['train'].time_series[k]['PW_0.5h'] for k in train_days
])
X = pd.concat([
    br['train'].time_series[k].drop(
        columns=['timestamp','PW_0.5h','date','time']
    ) for k in train_days
])

# Drop std=0 variables
X = X[
    [c for c in X.columns if np.std(X[c]) != 0]
]

keep_cols = X.columns

print(X.shape)
np.mean(X, axis=0)

# %%
def exceeds_5_breaches(values:np.ndarray, ucl):
    if (values > ucl).sum() >= 5:
        return True
    return False

# %%
chart = spc.FControlChart()

chart.determine_parameters(X.values)

exceeds_5per15 = spc.PatternFunction(
    exceeds_5_breaches,
    15,
    {'ucl': chart.ucl}
)

chart.add_patterns({'5per15at0.05':exceeds_5per15})

# %%
eval_keys = list(br['test'].time_series.keys())
k = eval_keys[0]
test_vals = br['test'].time_series[k][keep_cols].values
test_y = br['test'].time_series[k]['PW_0.5h']

hits = chart.check_patterns(test_vals)
hits

# %%
test_y.sum()
