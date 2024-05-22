# %%
import pathlib as plib
from tga_data_analysis.tga import Project, Sample

# this is the relative path to the folder where the data is
# which is the same as where the script is __file__
folder_path = plib.Path(__file__).resolve().parent
# if running as a Jupyter notebook, use absolute paths
# folder_path = r"absolute path to folder"
# %%
# create the project instance, important parameters are:
# the moisture value is used to compute the dry basis curve
proj_default = Project(
    folder_path,
    name="default",  # the name of the project
    temp_unit="C",  # the temperature that results will use (C or K)
    plot_font="Dejavu Sans",  # chose the font for the plots
    resolution_sec_deg_dtg=5,  # chose the resolution for dtg vectors
    dtg_window_filter=None,  # chose the filtering window for dtg curve
    plot_grid=False,  # wheter to include a grid in plots
    temp_initial_celsius=40,  # initial temperature for all curves (exclude data before)
    temp_lim_dtg_celsius=None,  # temperature limits for the dtg curves
    time_moist=38,  # the time where mass loss due to moisture is computed,
    time_vm=None,  # specify if there is a step to evaluate volatile matter
)
# add the first sample
cell = Sample(
    project=proj_default,  # specify what project it belongs to
    name="cell",  # sample name
    filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],  # filenames in the project folder
    time_moist=38,  # this was already specified in the project, but it can be overwritten
    time_vm=None,  # same as time_moist
)
# plot the tg_dtg plot for the cell sample (show all replicates to allow checks)
mf = cell.plot_tg_dtg()
# %%
# specify a difference project instance with different parameters
proj_mod = Project(
    folder_path,
    name="mod",
    temp_unit="K",
    plot_font="Times New Roman",
    resolution_sec_deg_dtg=2,  # bad choice, but shows the parameter use (see DTG (db))
    dtg_window_filter=51,  # smoothing window for dtg curve
    plot_grid=True,
    temp_initial_celsius=50,
    temp_lim_dtg_celsius=(150, 750),
    time_moist=38,
    time_vm=None,
)
# add a sample to the new project instance
cell = Sample(
    project=proj_mod,
    name="cell",
    filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
)
# plot the tg_dtg plot for the cell sample (show all replicates to allow checks)
mf = cell.plot_tg_dtg()
test= cell.ddtg_analysis()
# %%
import matplotlib.pyplot as plt
plt.plot(cell.temp_ddtg(), abs(cell.ddtg_db()))
plt.plot(cell.temp_dtg(),abs(cell.dtg_db()))
# %%
