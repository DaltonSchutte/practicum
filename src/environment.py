"""
Environment object for running experiments
"""
import os
import datetime
import random
from typing import (
    Optional,
    NoReturn,
    Union,
    Any
)

import numpy as np

from .data import TimeSeries


# ##########
# CLASSES #
# ##########

class TimeSeriesEnv:
    def __init__(self, timeseries: TimeSeries, feature_cols: list[str], label_col: str, shuffle_days: bool):
        super().__init__()
        self.timeseries = timeseries
        self.dates = sorted(self.timeseries.time_series.keys())
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.shuffle_days = shuffle_days

        if self.shuffle_days:
            random.shuffle(self.dates)

        # Track when the agent lets it run after it should have been stopped
        self.time_since_stopped = 0
        self.total_t = sum(len(ts) for ts in self.timeseries.time_series.values())

        # Reset will always be called immediately during training

    def reset(self):
        self.current_date_index = 0
        if self.shuffle_days:
            random.shuffle(self.dates)
        self.current_date = self.dates[self.current_date_index]
        self.current_time = 0
        self.state = self.timeseries\
                .time_series[self.current_date][self.feature_cols]\
                .iloc[self.current_time].values
        self.label = self.timeseries\
                .time_series[self.current_date][self.label_col]\
                .iloc[self.current_time]
        self.time_since_stopped = 0
        self.done = False
        self.day_done = False
        return self.state

    def forgetful_state_transition(self, action):
        if action==1:
            next_state = np.zeros((1,len(self.feature_cols)))
        else:
            if self.current_time+1 >= len(self.timeseries.time_series[self.current_date]):
                next_state = np.ones((1, len(self.feature_cols)))
            else:
                next_state = self.timeseries\
                        .time_series[self.current_date][self.feature_cols]\
                        .iloc[self.current_time+1]
        return next_state


    def step(self, action: int):
        if action==1:
            print('Date:  ', self.current_date,' stopped at: ', self.current_time)
            # The agent stops when the machine was stopped
            if self.label==1:
                reward = 1
                self.day_done = True
            # The agent stopped the machine early
            else:
                reward = -10
                self.day_done = True
        else:
            # The agent has not stopped the machine when it should have been
            if self.label==1:
                self.time_since_stopped += 0.1
                reward = -self.time_since_stopped
                print('Should have stopped at: ', self.current_time)
            # The agent does not stop the machine early
            else:
                reward = 0.01

        day_ended = self.day_done

        self.current_time += 1
        # We reached the end of the time series for that day or the agent stopped
        if (self.current_time >= len(self.timeseries.time_series[self.current_date])) \
           or (self.day_done):
            self.current_date_index += 1
            self.current_time = 0
            self.time_since_stopped = 0
            self.day_done = False

        # We reached the end of all the data
        if self.current_date_index >= len(self.timeseries.time_series):
            self.done = True
            # Agent gets a bonus for getting to the end without a false alarm
            if (action==0) and (self.label==0):
                reward = 1
            # It loses points for failing to stop when it should have
            if (action==0) and (self.label==1):
                reward = -1
            return self.state, reward, self.done, day_ended

        # Next state and label
        self.current_date = self.dates[self.current_date_index]
        if self.day_done:
            print(self.current_date)
        self.state = self.timeseries\
                .time_series[self.current_date][self.feature_cols]\
                .iloc[self.current_time].values
        self.label = self.timeseries\
                .time_series[self.current_date][self.label_col]\
                .iloc[self.current_time]
        return self.state, reward, self.done, day_ended
