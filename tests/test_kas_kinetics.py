# %%
import pytest
import pathlib as plib
import pandas as pd
import numpy as np
from tga_data_analysis.tga import Project, Sample
from tga_data_analysis.kas_kinetics import KasSample

threshold = 0.01


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
    checked_activation_energy = [
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
    for result, checked in zip(list(cell.activation_energy), checked_activation_energy):
        assert abs((result - checked) / checked) <= threshold
    checked_activation_energy_std = [
        62.95374239597605,
        58.931717137740215,
        53.73289953102068,
        51.33420981082763,
        48.19086595710156,
        46.50062643198576,
        44.42771515928734,
        42.06634599571246,
        41.49640293575315,
        39.015528074294735,
        38.26486459102316,
        37.70896291618733,
        34.74998368815841,
        32.152019180225814,
        19.70775940035246,
    ]
    for result, checked in zip(list(cell.activation_energy_std), checked_activation_energy_std):
        assert abs((result - checked) / checked) <= threshold
    checked_pre_exp_factors = [
        1.17932753e14,
        2.04512813e13,
        6.79099110e12,
        3.96979402e12,
        2.27422115e12,
        1.42941319e12,
        1.17555886e12,
        7.58394697e11,
        7.06855156e11,
        5.17262358e11,
        4.89148772e11,
        4.66487188e11,
        1.11427288e12,
        2.88682056e13,
        4.84937842e17,
    ]
    for result, checked in zip(list(cell.pre_exponential_factor), checked_pre_exp_factors):
        assert abs((result - checked) / checked) <= threshold


# %%
