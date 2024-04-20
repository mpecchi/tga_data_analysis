# %%
import pytest
import pathlib as plib
import pandas as pd
import numpy as np
from tga_data_analysis.tga import Measure, Sample, Project

threshold = 1e-5


# %%
@pytest.mark.parametrize("temp_symbol", ["C", "K"])
def test_deconvolution_with_temperature(test_dir, temp_symbol):

    proj = Project(test_dir, name="test", temp_unit=temp_symbol)

    misc = Sample(
        project=proj, name="misc", filenames=["MIS_1", "MIS_2", "MIS_3"], time_moist=38, time_vm=147
    )
    if temp_symbol == "K":
        misc.deconv_analysis([280 + 273.15, 380 + 273.15])
    else:
        misc.deconv_analysis([280, 380])

    assert abs(min(misc.dcv_best_fit()) + 1.0045394426179157) <= threshold


# %%
