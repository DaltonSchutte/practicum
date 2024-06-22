"""
Tools for handling the time-series data
"""
import os
from typing import (
    NoReturn,
    Optional,
    Any,
    Union
)

from tqdm.auto import tqdm

import pandas as pd
import polars as pl


###########
# GLOBALS #
###########

BACKENDS = [
    'polars',
    'pandas'
]


###########
# CLASSES #
###########

class TimeSeries:
    def __init__(
        data: Any,
        backend: Optional[str]='polars'
    )
    ):
        self._check_backend(backend)
        self.backend = backend

        self.data = self._prepare_time_series(
            data
        )

    def _prepare_time_series(
        self,
        data: Union[pd.DataFrame, pl.DataFrame]
    ):
        pass

    @staticmethod
    def _check_backend(
        backend: str
    ) -> NoReturn:
        if not backend in BACKENDS:
            msg = (
                f"Invalid backend: {backend}! "
                f"Use one of [{', '.join(BACKENDS)}]"
            )
            raise ValueError(msg)

    @classmethod
    def from_parquet(
        cls,
        backend: str,
        filepath: str,
        *args,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        self._check_backend(backend)

        if backend == 'polars':
            data = pl.read_parquet(
                filepath,
                *args,
                **kwargs
            )
        elif backend == 'pandas':
            data = pl.read_parquet(
                filepath,
                *args,
                **kwargs
            )
        return data


    @classmethod
    def from_csv(
        cls,
        backend: str,
        filepath: str,
        sep: Optional[str]='\t',
        *args,
        **kwargs
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        self._check_backend(backend)

        if backend == 'polars':
            data = pl.read_csv(
                filepath,
                separator=sep,
                *args,
                **kwargs
            )
        elif backend == 'pandas':
            data = pd.read_csv(
                filepath,
                sep=sep,
                *args,
                **kwargs
            )
        return cls(data)


#############
# FUNCTIONS #
#############


