import math

import numpy as np
import pandas as pd
import scipy as sp

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


#############
# FUNCTIONS #
#############

def mean_time_from_event(timeseries, stop_times, strict=False):
    diffs = []
    for ts, stop_time in zip(timeseries, stop_times):
        rng = [i for i, stop in enumerate(ts) if stop==1]
        if strict:
            # Measure from the earliest desireable stop time
            diffs.append(stop_time - min(rng))
        else:
            # Measure from the closer end of the interval
            if stop_time < min(rng):
                diffs.append(stop_time - min(rng))
            elif stop_time > max(rng):
                diffs.append(stop_time - max(rng))
            else:
                diffs.append(0)
    return np.mean(0)


def evaluate(timeseries, stop_times, strict=False):
    metrics = {}

    
