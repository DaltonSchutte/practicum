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
import datetime
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
pd.options.mode.chained_assignment = None
IMAGE_DIR = '/home/dalton/projects/practicum/deliverables/latex/assets/spc/'

# %%
ResultTup = namedtuple(
    'ResultTup',
    ['split_pct','pattern','strict','mean_time_from_event','f1','precision','recall']
)

def exceeds_n_breaches(values: np.ndarray, ucl, lcl, n):
    if ((values > ucl).sum() + (values<lcl).sum() )>= n:
        return True
    return False

def n_sequential_breaches(values: np.ndarray, ucl, lcl, n):
    for val in values:
        if (val > ucl) or (val < lcl):
            continue
        else:
            return False
    return True

# %%
data = {
    'blood-refrigerator': {'date': datetime.date(year=2022, month=12, day=26)},
    'nitrogen-generator': {"date": datetime.date(year=2023, month=9, day=29)}
}
for dir in os.listdir('../data'):
    if dir == 'wrapper-machine':
        continue
    if '.zip' in dir:
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
        c for c in temp.columns if temp[c].std()!=0
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
        BINARY_COLS = [
            c for c in X.columns if X[c].max()==1 and X[c].min()==0
        ]
        NONBINARY_FEATURE_COLS = [
            c for c in FEATURE_COLS if c not in BINARY_COLS
        ]
        X_tmp = X[BINARY_COLS]

        X = X[NONBINARY_FEATURE_COLS]
        X_mean = X.mean()
        X_max = X.max()
        X_min = X.min()
        X = pd.concat([X, X_tmp], axis=1)
            
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
        'X': {dt: x[FEATURE_COLS] for dt, x in test_ts.time_series.items()},
        'y': {dt: x['PW_0.5h'].values for dt, x in test_ts.time_series.items()}
    }
    test_data['X'] = {
        dt: pd.concat(
            [
                (x[NONBINARY_FEATURE_COLS]-X_mean)/(X_max-X_min)
            ] + [
                x[BINARY_COLS]
            ],
            axis=1
        ).values for dt, x in test_data['X'].items()
    }

    charts = {}

    for nm, split in train_splits.items():
        chart = FControlChart()
        try:
            chart.determine_parameters(split['X'])
        except np.linalg.LinAlgError:
            continue
        charts.update(
            {
                nm: chart
            }
        )
        print(nm, chart.lcl, chart.center_line, chart.ucl)

    test_matches = {}

    for nm, chart in charts.items():
        test_matches[nm] = {}

        for n in [5,10,20,30,40,60,120]:
            chart.add_patterns(
                {
                    f'{n}per{n*2}at0.05': PatternFunction(
                        exceeds_n_breaches,
                        int(n*2),
                        {'ucl': chart.ucl, 'n':n, 'lcl': chart.lcl}
                    ),
                    f'{n}seqAt0.05': PatternFunction(
                        exceeds_n_breaches,
                        int(n),
                        {'ucl': chart.ucl, 'n':n, 'lcl': chart.lcl}
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
            test_result_for_out.append(
                ResultTup(
                    pct,
                    pattern,
                    1,
                    mtfe,
                    mets['f1'],
                    mets['precision'],
                    mets['recall']
                )
            )
        
    test_result_df = pd.DataFrame(test_result_for_out)
    test_result_df.to_csv(
        os.path.join('../results/spc',dir,'test-results.tsv'),
        sep='\t',
        header=True,
        index=False
    )
    pickle.dump(
        charts,
        open(os.path.join('../results/spc',dir,'charts.pkl'), 'wb')
    )
    data.update(
        {
            dir: {
                'test': test_data,
                'FEATURES': FEATURE_COLS,
                'LABEL': LABEL_COL,
                'date': data[dir]['date'],
                'matches': test_matches
            }
        }
    )

# %%
(data['blood-refrigerator']['test']['y'][data['blood-refrigerator']['date']].shape,
data['nitrogen-generator']['test']['y'][data['nitrogen-generator']['date']].shape)

# %%
len(data['blood-refrigerator']['matches']['0.25'])

# %%
for nm, dset in data.items():
    print(nm.upper())

    charts = pickle.load(open(os.path.join('../results/spc',nm,'charts.pkl'),'rb'))


    for (dt, X), (_, y) in zip(dset['test']['X'].items(), dset['test']['y'].items()):
        print(dt)
        n = len(X)
        which_y = 0
        fig, axs = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=True,figsize=(10,10), layout='tight')

        ptns = set()
        for i, (pct, chart) in enumerate(charts.items()):
            which_x = i // 2
            Q = chart(X)
            ax = axs[which_x, which_y]

            # Q plot
            sns.lineplot(
                x=range(len(Q)),
                y=Q,
                ax=ax
            )
            # Red stop region
            try:
                start_idx = np.where(y==1)[0][0]
                end_idx = np.where(y==1)[0][-1]
                ax.axvspan(start_idx,end_idx, facecolor='red', alpha=0.25)
            except IndexError:
                pass

            # Chart stop line
            patterns = dset['matches'][pct]
            
            cmap = plt.cm.tab20
            cmap = [cmap(i) for i in range(len(patterns))]

            for i, (ptn, rs) in enumerate(patterns.items()):
                idx, stopped = rs[dt]

                if stopped:
                    if ptn in ptns:
                        label = '_nolegend_'
                    else:
                        label = ptn
                        ptns.add(ptn)
                    ax.axvline(idx, color=cmap[i], alpha=0.75, linestyle='--', label=label)

            # Chart parameters per percentage
            for ln, c in zip([chart.lcl, chart.center_line, chart.ucl],['red','green','red']):
                ax.axhline(
                    ln,
                    color=c
                )
            ax.set(
                title=f'Using {float(pct)*100:.0f}% of\nmost recent days',
                yscale='log',
                xlabel='Timestep',
                ylabel='Q values (log)'
            )
            which_y = int(which_y != 1)
        fig.suptitle(f'Control Charts for: {dt}')
        fig.legend()
        fig.savefig(
            os.path.join(IMAGE_DIR,nm,f'T-sq-cchart-{dt}.png'),
            dpi=400
        )
        plt.close()

# %%
# Daily re-eval
data = {
    'blood-refrigerator': {'date': datetime.date(year=2022, month=12, day=26)},
    'nitrogen-generator': {"date": datetime.date(year=2023, month=9, day=29)}
}
for dir in os.listdir('../data'):
    if dir == 'wrapper-machine':
        continue
    if '.zip' in dir:
        continue

    print(dir.upper())
    # Data loading
    data_dir_path = os.path.join('../data', dir)
    if dir == 'nitrogen-generator':
        tf = '%Y-%m-%d %H:%M:%S'
    else:
        tf = '%Y-%m-%d %H:%M:%S.%f'

    test_ts = TimeSeries.from_csv(
        'pandas',
        os.path.join(data_dir_path, 'test.csv')
    )

    # Data prep
    test_ts.parse_datetime('timestamp', tf)

    test_ts.split_by_day()

    temp = pd.concat(
        test_ts.time_series[k].drop(
            columns=['timestamp','PW_0.5h','date','time']
        ) for k in test_ts.time_series.keys()
    )

    FEATURE_COLS = [
        c for c in temp.columns if temp[c].std()!=0
    ]
    LABEL_COL = 'PW_0.5h'
    BINARY_COLS = [
        c for c in FEATURE_COLS if (temp[c].max()==1) and (temp[c].min()==0)
    ]
    NONBINARY_FEATURE_COLS = [
        c for c in FEATURE_COLS if not c in BINARY_COLS
    ]

    temp = None
    del temp

    charts = {}

    for dt, day_data in test_ts.time_series.items():
        y = day_data[LABEL_COL]
        X = day_data[FEATURE_COLS]
        X_tmp = X[BINARY_COLS]
        X = X[NONBINARY_FEATURE_COLS]
        X = (X-X.mean())/(X.max()-X.min())
        X = pd.concat(
            [X, X_tmp],
            axis=1
        )
        split_len = 150 if dir == 'blood-refrigerator' else 75
        X_fit = X[:split_len]
        X_eval = X[split_len:]
        X = X_fit
        chart = FControlChart()
        try:
            chart.determine_parameters(X.values)
        except np.linalg.LinAlgError:
            try:
                chart = charts[prev_dt]['chart']
            except KeyError:
                chart = None
        charts.update(
            {
                dt: {
                    'chart': chart,
                    'X_eval': X_eval,
                    'y': y
                }
            }
        )
        prev_dt = dt
        if chart is not None:
            print(dt, chart.lcl, chart.center_line, chart.ucl)
    # Back prop any missing
    for dt, stf in reversed(charts.items()):
        if stf['chart'] is None:
            charts[dt]['chart'] = prev_stf['chart']
        prev_stf = stf

    test_matches = {}

    for dt, comps in charts.items():
        print(dt)
        chart = comps['chart']
        X = comps['X_eval']
        test_matches[dt] = {}

        for n in [5,10,20,30,40,60,120]:
            chart.add_patterns(
                {
                    f'{n}per{n*2}at0.05': PatternFunction(
                        exceeds_n_breaches,
                        int(n*2),
                        {'ucl': chart.ucl, 'n':n, 'lcl': chart.lcl}
                    ),
                    f'{n}seqAt0.05': PatternFunction(
                        exceeds_n_breaches,
                        int(n),
                        {'ucl': chart.ucl, 'n':n, 'lcl': chart.lcl}
                    )
                }
            )

        matched = chart.check_patterns(X.values)
        print(X.values)
        print()
        for pattern, res in matched.items():
            if not test_matches[dt].get(pattern, False):
                test_matches[dt].update({pattern: {}})
            test_matches[dt][pattern].update(
                {
                    dt: res
                }
            )

    test_result_for_out = []

    # Non-strict eval
    all_ys = {
        dt: stf['y'].values for dt,stf in charts.items()
    }
    # Re-structure
    matches_rest = {}
    for dt, res in test_matches.items():
        for ptn, matches in res.items():
            if not matches_rest.get(ptn, False):
                matches_rest.update({ptn: {dt: matches[dt]}})
            else:
                matches_rest[ptn].update({dt: matches[dt]})

    for pattern, matches in matches_rest.items():  
        diffs, mtfe = mean_time_from_event(all_ys, matches)
        hits, mets = classification_metrics(all_ys, matches)

        test_result_for_out.append(
            ResultTup(
                dt,
                pattern,
                0,
                mtfe,
                mets['f1'],
                mets['precision'],
                mets['recall']
            )
        )

    # Strict eval
    for pattern, matches in matches_rest.items():  
        diffs, mtfe = mean_time_from_event(all_ys, matches, strict=True)
        hits, mets = classification_metrics(all_ys, matches, strict=True)
        test_result_for_out.append(
            ResultTup(
                dt,
                pattern,
                1,
                mtfe,
                mets['f1'],
                mets['precision'],
                mets['recall']
            )
        )
        
    test_result_df = pd.DataFrame(test_result_for_out)
    test_result_df.to_csv(
        os.path.join('../results/spc/daily',dir,'test-results.tsv'),
        sep='\t',
        header=True,
        index=False
    )
    pickle.dump(
        charts,
        open(os.path.join('../results/spc/daily',dir,'charts.pkl'), 'wb')
    )
    data.update(
        {
            dir: {
                'FEATURES': FEATURE_COLS,
                'LABEL': LABEL_COL,
                'date': data[dir]['date'],
                'matches': matches_rest
            }
        }
    )

# %%
for nm, dset in data.items():
    print(nm.upper())

    charts = pickle.load(open(os.path.join('../results/spc/daily',nm,'charts.pkl'),'rb'))

    which_y = 0
    fig = plt.figure(
        figsize=(10,10),
        layout='tight'
    )

    ptns = set()
    for i, (dt, stf) in enumerate(charts.items()):
        fig, axs = plt.subplots(
            nrows=1,
            ncols=1,
            sharex=False,
            sharey=False,
            figsize=(5,5),
            layout='tight'
        )
        which_x = 0

        print(dt, which_x, which_y)
        chart = stf['chart']
        X = stf['X_eval'].values
        y = stf['y']
        Q = chart(stf['X_eval'].values)
        ax = axs

        # Q plot
        sns.lineplot(
            x=range(len(Q)),
            y=Q,
            ax=ax
        )
        # Red stop region
        try:
            start_idx = np.where(y==1)[0][0]
            end_idx = np.where(y==1)[0][-1]
            ax.axvspan(start_idx,end_idx, facecolor='red', alpha=0.25)
        except IndexError:
            pass

        patterns = dset['matches']
        # Chart stop line

        cmap = plt.cm.tab20
        cmap = [cmap(i) for i in range(len(patterns))]

        for i, (ptn, rs) in enumerate(patterns.items()):
            idx, stopped = rs[dt]

            if stopped:
                if ptn in ptns:
                    label = '_nolegend_'
                else:
                    label = ptn
                    ptns.add(ptn)
                ax.axvline(idx, color=cmap[i], alpha=0.75, linestyle='--', label=label)

        # Chart parameters per percentage
        for ln, c in zip([chart.lcl, chart.center_line, chart.ucl],['red','green','red']):
            ax.axhline(
                ln,
                color=c
            )
        ax.set(
            title=f'Date: {dt}',
            yscale='log',
            xlabel='Timestep',
            ylabel='Q values (log)'
        )
        fig.suptitle(f'Control Charts for: {dt}')
        fig.legend()
#        fig.savefig(
#            os.path.join(IMAGE_DIR,nm,f'T-sq-cchart-{dt}.png'),
#            dpi=400
#        )
#        plt.close()


