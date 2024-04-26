# %%
import pathlib as plib
from tga_data_analysis.tga import Project, Sample

# this is the relative path to the folder where the data is
# which is the same as where the script is __file__
folder_path = plib.Path(__file__).resolve().parent
# if running as a Jupyter notebook, use absolute paths
# folder_path = plib.Path("absolute path to folder")
# %%
# create the project instance, important parameters are:
# temp_unit: the temperature that results will use
# soliddist_steps_min: the end of the steps (in min) where the mass loss
# for the step must be computed.
# time_moist: the time where mass loss due to moisture is computed,
# the moisture value is used to compute the dry basis curve
# time_vm: set to ignore volatile matter computation for soliddist
proj_soliddist = Project(
    folder_path,
    temp_unit="K",
    soliddist_steps_min=[40, 70, 100, 130, 160, 190],
    time_moist=38,
    time_vm=None,
)
# add the first sample, specify the sample name and the filenames
sda = Sample(
    project=proj_soliddist,
    name="sda",
    filenames=["SDa_1", "SDa_2", "SDa_3"],
)
# add the second sample, specify the sample name and the filenames
sdb = Sample(
    project=proj_soliddist,
    name="sdb",
    filenames=["SDb_1", "SDb_2", "SDb_3"],
)
# create and save soliddist reports for each sample (with each replicate)
repa = sda.report("soliddist")
repb = sdb.report("soliddist")
# plot the soliddist curves for each sample (showing replicates)
mf = sda.plot_soliddist()
mf = sdb.plot_soliddist()
# create a multireports with samples side by side
rep = proj_soliddist.multireport(report_type="soliddist")
# plot the soliddist curve of all samples side by side
mf = proj_soliddist.plot_multi_soliddist()
# plot the report for all samples with the step mass loss
mf = proj_soliddist.multireport(report_type="soliddist")
# %%
