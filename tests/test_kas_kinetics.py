# %%
import pytest
import pathlib as plib
import pandas as pd
import numpy as np
from tga_data_analysis.tga import Project, Sample
from tga_data_analysis.kas_kinetics import KasSample

threshold = 1e-5


# %%
@pytest.mark.parametrize("temp_symbol", ["C", "K"])
def test_kas_kinetics_with_temperature(test_dir, temp_symbol):

    proj = Project(folder_path=test_dir, name="test", temp_unit=temp_symbol)
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
    cell_ox50 = Sample(
        project=proj,
        name="cell_ox50",
        load_skiprows=8,
        filenames=["CLSOx50_4", "CLSOx50_5", "CLSOx50_6"],
        time_moist=38,
        time_vm=None,
        heating_rate_deg_min=50,
    )
    cell_ox100 = Sample(
        project=proj,
        name="cell_ox100",
        load_skiprows=8,
        filenames=["CLSOx100_4", "CLSOx100_5", "CLSOx100_6"],
        time_moist=38,
        time_vm=None,
        heating_rate_deg_min=100,
    )
    cell = KasSample(proj, samples=[cell_ox5, cell_ox10, cell_ox50, cell_ox100], name="cellulose")
    cell.kas_analysis()
    checked_results = [
        182.98695498901722,
        177.93533544211007,
        175.00305447358866,
        174.24922678890448,
        173.10247795834562,
        172.17212058541972,
        172.38983956887415,
        171.32000094274244,
        171.9362038980991,
        171.33484227896233,
        171.9242539930856,
        172.52779644872686,
        177.92903770087554,
        195.65939854731468,
        247.71378828071303,
    ]
    for result, checked_result in zip(list(cell.activation_energy), checked_results):
        assert abs(result - checked_result) <= threshold


# %%
