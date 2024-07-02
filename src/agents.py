"""
Class that facilitates interaction with the environment
"""
from typing import (
    Optional,
    NoReturn,
    Any,
    Callable,
    Union
)

import numpy as np

from spc import ControlChartBase


###########
# CLASSES #
###########

class Agent:
    def __init__(
        self,
        method: Union[ControlChartBase],
        gamma: float,
        epsilon: Optional[float]=None,
        epsilon_decay: Optional[float]=None,
        tau: Optional[float]=None
    ):
        pass

    def act(self, state: np.ndarray):
        pass

    def learn(self):
        pass

    def update(self):
        pass
