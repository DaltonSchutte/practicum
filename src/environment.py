"""
Environment object for running experiments
"""
import os
import datetime
from typing import (
    Optional,
    NoReturn,
    Union,
    Any
)

from tqdm.auto import tqdm

import numpy as np

from data import TimeSeries


###########
# CLASSES #
###########

class Environment(EnvBase):
    def __init__(
        self,
        series: TimeSeries,
        seed: Optional[int]=3141
    ):
        self.series = series

        self.date_order = list(series.time_series.keys())
        self.date_order.sort()
        self.t = 0
        self.curr_date = self.date_order[self.t]


    def step(self, action: int):
        pass

    def _reward(self):
        pass

    def reset(self):
        self.t = 0
        self.curr_date = self.date_order[self.t]

    def set_seed(self):
        pass
