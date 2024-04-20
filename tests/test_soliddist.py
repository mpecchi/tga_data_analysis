# %%
import pytest
import pathlib as plib
import pandas as pd
import numpy as np
from tga_data_analysis.tga import Measure, Sample, Project

threshold = 1e-5


# %%
@pytest.mark.parametrize("temp_symbol", ["C", "K"])
def test_soliddist_with_temperature(test_dir, temp_symbol):

    proj = Project(test_dir, name="test", temp_unit=temp_symbol)
    sda = Sample(
        project=proj,
        name="sda",
        filenames=["SDa_1", "SDa_2", "SDa_3"],
        time_moist=38,
        time_vm=None,
    )
    sdb = Sample(
        project=proj,
        name="sdb",
        filenames=["SDb_1", "SDb_2", "SDb_3"],
        time_moist=38,
        time_vm=None,
    )
    sda.soliddist_analysis()
    sdb.soliddist_analysis()
    sda_dmp_array = [0.0, 0.41662584, 50.2045787, 36.57805269, 2.03404787, 0.81554841, 9.93150094]
    for result, checked_value in zip(list(sda.dmp_soliddist()), sda_dmp_array):
        assert abs(result - checked_value) <= threshold


# %%
