# %%
import pytest
import pathlib as plib
import pandas as pd
import numpy as np
from tga_data_analysis.tga import Measure, Sample, Project


# %%
@pytest.mark.parametrize("temp_symbol", ["C", "K"])
def test_proximate_with_temperature(test_dir, temp_symbol):

    proj = Project(test_dir, name="test", temp_unit=temp_symbol)
    sru = Sample(
        project=proj, name="sru", filenames=["SRU_1", "SRU_2", "SRU_3"], time_moist=38, time_vm=167
    )
    misc = Sample(
        project=proj, name="misc", filenames=["MIS_1", "MIS_2", "MIS_3"], time_moist=38, time_vm=147
    )
    assert sru.moist_ar.stk() == {
        0: 0.20502144083498308,
        1: 0.2616042371125502,
        2: 0.31114554215777446,
    }
    assert sru.ash_db.stk() == {0: 1.3564700337049267, 1: 1.3336811418433068, 2: 1.4340140863270139}
    assert sru.vm_db.stk() == {0: 85.20223937085098, 1: 85.3162980049972, 2: 83.68320591127834}
    assert sru.fc_db.stk() == {0: 13.441290595444098, 1: 13.350020853159497, 2: 14.882780002394652}
    assert misc.moist_ar.stk() == {0: 6.21918135439951, 1: 6.341669304632262, 2: 6.920376944816013}
    assert misc.ash_db.stk() == {0: 0.0, 1: 0.0, 2: 0.0}
    assert misc.vm_db.stk() == {0: 83.6147296412197, 1: 84.20178008111762, 2: 83.5116153430578}
    assert misc.fc_db.stk() == {0: 16.385270358780293, 1: 15.79821991888238, 2: 16.488384656942195}


# %%
