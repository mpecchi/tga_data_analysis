# %%
import pytest
import numpy as np
from tga_data_analysis.tga import Sample, Project


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
    sru.proximate_analysis()
    misc.proximate_analysis()
    for to_check, checked in zip(
        sru.moist_ar.stk().values(), [0.20502144083498308, 0.2616042371125502, 0.31114554215777446]
    ):
        assert np.isclose(to_check, checked)

    for to_check, checked in zip(
        sru.ash_db.stk().values(), [1.3564700337049267, 1.3336811418433068, 1.4340140863270139]
    ):
        assert np.isclose(to_check, checked)

    for to_check, checked in zip(
        sru.vm_db.stk().values(), [85.20223937085098, 85.3162980049972, 83.68320591127834]
    ):
        assert np.isclose(to_check, checked)

    for to_check, checked in zip(
        sru.fc_db.stk().values(), [13.441290595444098, 13.350020853159497, 14.882780002394652]
    ):
        assert np.isclose(to_check, checked)

    for to_check, checked in zip(
        misc.moist_ar.stk().values(), [6.21918135439951, 6.341669304632262, 6.920376944816013]
    ):
        assert np.isclose(to_check, checked)

    for to_check, checked in zip(misc.ash_db.stk().values(), [0.0, 0.0, 0.0]):
        assert np.isclose(to_check, checked)

    for to_check, checked in zip(
        misc.vm_db.stk().values(), [83.6147296412197, 84.20178008111762, 83.5116153430578]
    ):
        assert np.isclose(to_check, checked)

    for to_check, checked in zip(
        misc.fc_db.stk().values(), [16.385270358780293, 15.79821991888238, 16.488384656942195]
    ):
        assert np.isclose(to_check, checked)


# %%
