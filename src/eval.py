"""
Evaluation tools
"""
import datetime
from typing import (
    Any,
    Optional
)

import numpy as np
import pandas as pd
import scipy as sp

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


#############
# FUNCTIONS #
#############

def mean_time_from_event(
    timeseries: dict[datetime.datetime, Any], 
    stop_times: dict[datetime.datetime, Any], 
    strict: Optional[bool]=False
) -> tuple[dict[datetime, int], float]:
    diffs = {}
    for date, ts in timeseries.items():
        stop_time, stopped = stop_times[date]
        if 1 not in ts:
            if stopped:
                diffs.update({date: stop_time})
                continue
            else:
                diffs.update({date: 0})
                continue
        else:
            if not stopped:
                stop_time = ts[-1]

        rng = [i for i, stop in enumerate(ts) if stop==1]
        if strict:
            # Measure from the earliest desireable stop time
            diff = stop_time - min(rng)
        else:
            # Measure from the closer end of the interval
            if stop_time < min(rng):
                diff = stop_time - min(rng)
            elif stop_time > max(rng):
                diff = stop_time - max(rng)
            else:
                diff = 0
        diffs.update({date: diff})
    return (diffs, np.mean(list(diffs.values())))


def classification_metrics(
    timeseries: dict[datetime.datetime, Any], 
    stop_times: dict[datetime.datetime, Any], 
    strict: Optional[bool]=False
) -> tuple[dict[str, tuple[int,int]], dict[str, float]]:
    hits = {}
    for date, ts in timeseries.items():
        stop_time, stopped = stop_times[date]
        rng = [i for i, stop in enumerate(ts) if stop==1]
        if 1 not in ts:
            if stopped:
                hits.update({date: (1,0)})
                continue
            else:
                hits.update({date: (1,1)})
                continue
        else:
            if not stopped:
                hits.update({date: (1,0)})
        if strict:
            hits.update(
                {
                    date: (
                        ts[stop_time],
                        1 if min(rng)==stop_time else 0
                    )
                }
            )
        else:
            hits.update(
                {
                    date: (
                        ts[stop_time],
                        1 if stop_time in rng else 0
                    )
                }
            )
    y_true=[y[0] for y in hits.values()]
    y_pred=[y[1] for y in hits.values()]
    metrics = {
        'f1': f1_score(y_true, y_pred, zero_division=0.0),
        'recall': recall_score(y_true, y_pred, zero_division=0.0),
        'precision': precision_score(y_true, y_pred, zero_division=0.0)
    }
    return (hits, metrics)
