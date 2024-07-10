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
from string import Template
from collections import namedtuple

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '..')
from src.data import TimeSeries
from src.methods.spc import FControlChart, PatternFunction
from src.eval import (
    mean_time_from_event,
    classification_metrics
)

# %%
SEED = 3141

# %%
ResultTup = namedtuple(
    'ResultTup',
    ['split_pct','pattern','strict','mean_time_from_event','f1','precision','recall']
)

def exceeds_n_breaches(values: np.ndarray, ucl, n):
    if (values > ucl).sum() >= 5:
        return True
    return False

def n_sequential_breaches(values: np.ndarray, ucl, n):
    if (values > ucl).sum() == n:
        return True
    return False

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
    test_ts = TimeSeries.from_csv(
        'pandas',
        os.path.join(data_dir_path, 'test.csv')
    )

    # Data prep
    train_ts.parse_datetime('timestamp', tf)
    test_ts.parse_datetime('timestamp', tf)

    train_ts.split_by_day()
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

    splits = [0.25,0.5,0.75,1.0]
    train_splits = {}
    for pct in splits:
        n_days = len(train_ts.time_series)
        train_days = list(train_ts.time_series.keys())[-int(pct*n_days):]
        y = pd.concat([
            train_ts.time_series[k]['PW_0.5h'] for k in train_days
        ])
        X = pd.concat([
            train_ts.time_series[k].drop(
                columns=['timestamp','PW_0.5h','date','time']
            ) for k in train_days
        ])

        # Drop std=0 variables
        X = X[FEATURE_COLS]
            
        train_splits.update(
            {
                str(pct):
                {
                    'X': X.values,
                    'y': y.values
                }
            }
        )

    test_data = {
        'X': {dt: x[FEATURE_COLS].values for dt, x in test_ts.time_series.items()},
        'y': {dt: x['PW_0.5h'].values for dt, x in test_ts.time_series.items()}
    }

    charts = {}

    for nm, split in train_splits.items():
        chart = FControlChart()
        chart.determine_parameters(split['X'])
        charts.update(
            {
                nm: chart
            }
        )

    test_matches = {}

    for nm, chart in charts.items():
        test_matches[nm] = {}

        for n in [5,10,20,40,80]:
            chart.add_patterns(
                {
                    f'{n}per{n*2}at0.05': PatternFunction(
                        exceeds_n_breaches,
                        int(n*2),
                        {'ucl': chart.ucl, 'n':n}
                    ),
                    f'{n}seqAt0.05': PatternFunction(
                        exceeds_n_breaches,
                        int(n),
                        {'ucl': chart.ucl, 'n':n}
                    )
                }
            )

        for dt, X in test_data['X'].items():
            matched = chart.check_patterns(X)
            for pattern, res in matched.items():
                if not test_matches[nm].get(pattern, False):
                    test_matches[nm].update({pattern: {}})
                test_matches[nm][pattern].update(
                    {
                        dt: res
                    }
                )

    test_result_for_out = []

    # Non-strict eval
    for pct, res in test_matches.items():
        for pattern, matches in res.items():  
            diffs, mtfe = mean_time_from_event(test_data['y'], matches)
            hits, mets = classification_metrics(test_data['y'], matches)

            test_result_for_out.append(
                ResultTup(
                    pct,
                    pattern,
                    0,
                    mtfe,
                    mets['f1'],
                    mets['precision'],
                    mets['recall']
                )
            )

    # Strict eval
    for pct, res in test_matches.items():
        for pattern, matches in res.items():  
            diffs, mtfe = mean_time_from_event(test_data['y'], matches, strict=True)
            hits, mets = classification_metrics(test_data['y'], matches, strict=True)
            ResultTup(
                pct,
                pattern,
                1,
                mtfe,
                mets['f1'],
                mets['precision'],
                mets['recall']
            )
        
    test_result_df = pd.DataFrame(test_result_for_out)
    test_result_df.to_csv(
        os.path.join('../results/spc',dir,'test-results.tsv'),
        sep='\t',
        header=True,
        index=False
    )
