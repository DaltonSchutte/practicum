"""
Classes for statistical process control based techniques
"""
from typing import (
    Optional,
    NoReturn,
    Any,
    Callable
)

import numpy as np
from scipy import stats


###########
# CLASSES #
###########

class PatternFunction:
    def __init__(self, func, t: int, fargs: dict):
        self.func = func
        self.t = t
        self.fargs = fargs

    def __call__(self, x):
        windows = []
        for i in range(len(x)):
            windows.append(
                x[i:(i+self.t)]
            )
            if i+self.t+1 == len(x):
                break

        for i, window in enumerate(windows):
            is_match = self.func(window, **self.fargs)

            if is_match:
                return (i, True)
        return (-1, False)


class ControlChartBase:
    def __init__(
        self,
    ):
        self.patterns = {}

    def add_patterns(self, patterns: dict[str,PatternFunction]):
        self.patterns.update(patterns)

    def determine_parameters(self, data):
        raise NotImplementedError("Inhereting classes must overwrite!")

    def __call__(self, x):
        raise NotImplementedError("Inhereting classes must overwrite!")

    def check_patterns(self, x):
        raise NotImplementedError("Inhereting classes must overwrite!")


class PCAChart(ControlChartBase):
    def __init__(self, patterns, stop_criterion, pca):
        super().__init__(patterns, stop_criterion)
        pass

    def determine_parameters(self, data):
        pass

    def __call__(self, x):
        pass

    def check_patterns(self, x):
        pass


class FControlChart(ControlChartBase):
    def __init__(self):
        super().__init__()
        self.fitted = False

    def determine_parameters(self, X, alpha=0.05):
        Q = np.zeros(np.size(X, 0))
        self.mu = np.mean(X, axis=0)
        cov = np.cov(X.T)
        self.cov_inv = np.linalg.pinv(cov)
        obs = np.size(X, 0)
        for i in range(obs):
            spread = X[i,:] - self.mu
            Q[i] = np.matmul(np.matmul(spread, self.cov_inv), spread)

        self.Q_train = Q
        self.center_line = self.get_control_limit(0.5, X.shape[1])
        self.ucl = self.get_control_limit(1-alpha/2, X.shape[1])
        self.lcl = self.get_control_limit(alpha/2, X.shape[1])

        self.fitted = True

    def get_control_limit(self, x, n_vars):
        N = self.Q_train.shape[0]
        a = n_vars / 2
        b = (N - n_vars - 1) / 2
        return ((N-1)**2 /N ) * stats.beta.ppf(x, a, b)

    def __call__(self, X):
        Q = np.zeros(np.size(X, 0))
        obs = np.size(X, 0)
        for i in range(obs):
            spread = X[i,:]-self.mu
            Q[i] = np.matmul(np.matmul(spread, self.cov_inv), spread)
        return Q

    def check_patterns(self, X):
        """
        Only keeps first match
        """
        if not self.fitted:
            raise ValueError('Must fit with determine_parameters method first!')

        Q = self(X)
        matches = {}

        for pname, pfunc in self.patterns.items():
            results = pfunc(Q)
            matches.update(
                {pname: results}
            )

        return matches
