import pytest
import numpy as np
import pandas as pd
from typing import Literal
from tga_data_analysis.tga import Measure

# Assuming the Measure class is defined here or imported


def test_measure_initialization():
    measure = Measure("test_measure")
    assert measure.name == "test_measure"
    assert measure._stk == {}
    assert measure._ave is None
    assert measure._std is None


def test_add_single_value():
    measure = Measure()
    measure.add(1, 5.0)
    assert measure._stk[1] == 5.0


def test_add_array_value():
    measure = Measure()
    arr = np.array([1, 2, 3])
    measure.add(1, arr)
    assert np.array_equal(measure._stk[1], arr)


def test_add_series_value():
    measure = Measure()
    series = pd.Series([1, 2, 3])
    measure.add(2, series)
    assert np.array_equal(measure._stk[2], series.to_numpy())


def test_ave_single_value():
    measure = Measure()
    measure.add(1, 5.0)
    measure.add(2, 15.0)
    assert measure.ave() == 10.0


def test_ave_array_value():
    measure = Measure()
    measure.add(1, np.array([1, 2, 3]))
    measure.add(2, np.array([4, 5, 6]))
    assert np.array_equal(measure.ave(), np.array([2.5, 3.5, 4.5]))


def test_std_population():
    Measure.set_std_type("population")
    measure = Measure()
    measure.add(1, 10)
    measure.add(2, 20)
    assert measure.std() == 5.0


def test_std_sample():
    Measure.set_std_type("sample")
    measure = Measure()
    measure.add(1, 10)
    measure.add(2, 20)
    assert measure.std() == np.std([10, 20], ddof=1)


def test_set_std_type():
    Measure.set_std_type("population")
    assert Measure.std_type == "population"
    assert Measure.np_ddof == 0

    Measure.set_std_type("sample")
    assert Measure.std_type == "sample"
    assert Measure.np_ddof == 1
