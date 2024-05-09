# %%
import pytest
import pathlib as plib
import pandas as pd
import numpy as np
from tga_data_analysis.tga import Measure, Sample, Project

threshold = 1e-5


# %%
@pytest.mark.parametrize("temp_symbol", ["C", "K"])
def test_oxidation_with_temperature(test_dir, temp_symbol):
    proj = Project(folder_path=test_dir, name="test", temp_unit=temp_symbol, dtg_window_filter=101)
    cell_ox5 = Sample(
        project=proj,
        name="cell_ox5",
        filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
        time_moist=38,
        time_vm=None,
        heating_rate_deg_min=5,
    )
    cell_ox10 = Sample(
        project=proj,
        name="cell_ox10",
        load_skiprows=8,
        filenames=["CLSOx10_2", "CLSOx10_3", "CLSOx10_4"],
        time_moist=38,
        time_vm=None,
        heating_rate_deg_min=10,
    )
    cell_ox5.oxidation_analysis()
    cell_ox10.oxidation_analysis()
    if temp_symbol == "K":
        conv = +273.15
    else:
        conv = 0
    np.testing.assert_allclose(cell_ox5.temp_i.ave(), 270.18666666666667 + conv)
    np.testing.assert_allclose(cell_ox5.temp_b.ave(), 512.0 + conv)
    np.testing.assert_allclose(cell_ox10.temp_i.ave(), 276.1666666666667 + conv)
    np.testing.assert_allclose(cell_ox10.temp_b.ave(), 523.7833333333333 + conv)


# %%
