# %%
import pathlib as plib
from tga_data_analysis.tga import Project, Sample

test_dir = plib.Path(__file__).resolve().parent / "deconvolution"

proj = Project(test_dir, name="test", temp_unit="K")
cell = Sample(
    project=proj,
    name="cell",
    filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
    time_moist=38,
    time_vm=None,
)
misc = Sample(
    project=proj, name="misc", filenames=["MIS_1", "MIS_2", "MIS_3"], time_moist=38, time_vm=147
)
misc.deconv_analysis([280 + 273, 380 + 273])
cell.deconv_analysis([310 + 273, 450 + 273, 500 + 273])
mf = misc.plot_deconv()
mf = cell.plot_deconv()
# %%
misc2 = Sample(
    project=proj, name="misc2", filenames=["MIS_1", "MIS_2", "MIS_3"], time_moist=38, time_vm=147
)
misc2.deconv_analysis([250 + 273, 350 + 273, 410 + 273])
mf = misc2.plot_deconv()
# %%
