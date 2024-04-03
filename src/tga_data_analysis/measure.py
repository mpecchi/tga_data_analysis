from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal


class Measure:
    """
    A class to handle and analyze a series of measurements or data points. It provides functionalities
    to add new data, compute averages, and calculate standard deviations, supporting the analysis
    of replicated measurement data.
    """

    std_type: Literal["population", "sample"] = "population"
    if std_type == "population":
        np_ddof: int = 0
    elif std_type == "sample":
        np_ddof: int = 1

    @classmethod
    def set_std_type(cls, new_std_type: Literal["population", "sample"]):
        """
        Set the standard deviation type for all instances of Measure.

        This class method configures whether the standard deviation calculation should be
        performed as a sample standard deviation or a population standard deviation.

        :param new_std_type: The type of standard deviation to use ('population' or 'sample').
        :type new_std_type: Literal["population", "sample"]
        """
        cls.std_type = new_std_type
        if new_std_type == "population":
            cls.np_ddof: int = 0
        elif new_std_type == "sample":
            cls.np_ddof: int = 1

    def __init__(self, name: str | None = None):
        """
        Initialize a Measure object to store and analyze data.

        :param name: An optional name for the Measure object, used for identification and reference in analyses.
        :type name: str, optional
        """
        self.name = name
        self._stk: dict[int : np.ndarray | float] = {}
        self._ave: np.ndarray | float | None = None
        self._std: np.ndarray | float | None = None

    def __call__(self):
        return self.ave()

    def add(self, replicate: int, value: np.ndarray | pd.Series | float | int) -> None:
        """
        Add a new data point or series of data points to the Measure object.

        :param replicate: The identifier for the replicate to which the data belongs.
        :type replicate: int
        :param value: The data point(s) to be added. Can be a single value or a series of values.
        :type value: np.ndarray | pd.Series | float | int
        """
        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.to_numpy()
        elif isinstance(value, np.ndarray):
            value = value.flatten()
        elif isinstance(value, (list, tuple)):
            value = np.asarray(value)
        self._stk[replicate] = value

    def stk(self, replicate: int | None = None) -> np.ndarray | float:
        """
        Retrieve the data points for a specific replicate or all data if no replicate is specified.

        :param replicate: The identifier for the replicate whose data is to be retrieved. If None, data for all replicates is returned.
        :type replicate: int, optional
        :return: The data points for the specified replicate or all data.
        :rtype: np.ndarray | float
        """
        if replicate is None:
            return self._stk
        else:
            return self._stk.get(replicate)

    def ave(self) -> np.ndarray:
        """
        Calculate and return the average of the data points across all replicates.

        :return: The average values for the data points.
        :rtype: np.ndarray
        """
        if all(isinstance(v, np.ndarray) for v in self._stk.values()):
            self._ave = np.mean(np.column_stack(list(self._stk.values())), axis=1)
            return self._ave
        else:
            self._ave = np.mean(list(self._stk.values()))
            return self._ave

    def std(self) -> np.ndarray:
        """
        Calculate and return the standard deviation of the data points across all replicates.

        :return: The standard deviation of the data points.
        :rtype: np.ndarray
        """
        if all(isinstance(v, np.ndarray) for v in self._stk.values()):
            self._std = np.std(
                np.column_stack(list(self._stk.values())), axis=1, ddof=Measure.np_ddof
            )
            return self._std
        else:
            self._std = np.std(list(self._stk.values()), ddof=Measure.np_ddof)
            return self._std
