# %%
import pathlib as plib
from tga_data_analysis.tga import Project, Sample

folder_path = plib.Path(__file__).resolve().parent
# %%
proj_soliddist = Project(folder_path, name="test", temp_unit="K")

sda = Sample(
    project=proj_soliddist,
    name="sda",
    filenames=["SDa_1", "SDa_2", "SDa_3"],
    time_moist=38,
    time_vm=None,
)
sdb = Sample(
    project=proj_soliddist,
    name="sdb",
    filenames=["SDb_1", "SDb_2", "SDb_3"],
    time_moist=38,
    time_vm=None,
)
repa = sda.report("soliddist")
repb = sdb.report("soliddist")
mf = sda.plot_soliddist()
mf = sdb.plot_soliddist()

rep = proj_soliddist.multireport(report_type="soliddist")
mf = proj_soliddist.plot_multi_soliddist(labels=["sample 1", "sample 2", "sample 3"])
