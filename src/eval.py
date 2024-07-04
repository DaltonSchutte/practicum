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
    for (date, ts), stop_time in zip(timeseries.items(), stop_times):
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
        diffs.update{{date: diff}}
    return (diffs, np.mean(diffs.values))


def classification_metrics(
    timeseries: dict[datetime.datetime, Any], 
    stop_times: dict[datetime.datetime, Any], 
    strict: Optional[bool]=False
) -> tuple[dict[str, tuple[int,int]], dict[str, float]]:
    hits = {}
    for (date, ts), stop_time in zip(timeseries.items(), stop_times):
        rng = [i for i, stop in enumerate(ts) if stop==1]
        if strict:
            hits.update(
                {
                    date: (
                        ts[stop_time],,
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
    y_true=[y[0] for y in hits.values()],
    y_pred=[y[1] for y in hits.values()]
    metrics = {
        'f1': f1_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred)
    }
    return (hits, metrics)
