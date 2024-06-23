"""
Tools for handling the time-series data
"""
import os
import datetime
from typing import (
    NoReturn,
    Optional,
    Any,
    Union
)

from tqdm.auto import tqdm

import pandas
import polars


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
        self,
        data: Any,
        backend: Optional[str]='polars'
    ):
        self._check_backend(backend)
        self.backend = backend

        self.data = data

    @property
    def shape(self) -> tuple[int,int]:
        return self.data.shape

    def head(self, n: Optional[int]=5):
        return self.data.head(n)

    def _prepare_time_series(
        self,
        data: Union[pandas.DataFrame, polars.DataFrame]
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
    ) -> Union[pandas.DataFrame, polars.DataFrame]:
        cls._check_backend(backend)

        data = eval(backend).read_parquet(
            filepath,
            *args,
            **kwargs
        )
        return cls(data, backend)


    @classmethod
    def from_csv(
        cls,
        backend: str,
        filepath: str,
        *args,
        **kwargs
    ) -> Union[pandas.DataFrame, polars.DataFrame]:
        cls._check_backend(backend)

        data = eval(backend).read_csv(
            filepath,
            *args,
            **kwargs
        )
        return cls(data, backend)

    def split_by_day(
        self,
    ) -> NoReturn:
        self.time_series = {
            date: self.data[self.data['date']==date] for date in self.data['date'].unique()
        }

    def parse_datetime(
        self,
        column: str,
        datetime_format: Optional[str]='%Y-%m-%d %H:%M:%S.%f'
    ) -> NoReturn:
        self.data[['date','time']] = eval(self.backend).DataFrame(
            self.data[column].apply(parse_datetime).tolist()
        )


#############
# FUNCTIONS #
#############

def parse_datetime(
    string: str,
    datetime_format: Optional[str]='%Y-%m-%d %H:%M:%S.%f'
) -> tuple[datetime.date, datetime.time]:
    """
    Splits string into date and time objects

    :param string: text containing date and time data
    :param datetime_format: datetime format expected in string
    :return: tuple of the date and time as datetime objects
    """
    stripped = datetime.datetime.strptime(string, datetime_format)
    return (
        stripped.date(),
        stripped.time()
    )
