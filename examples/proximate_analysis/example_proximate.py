# %%
import pathlib as plib
from tga_data_analysis.tga import Project, Sample

# this is the relative path to the folder where the data is
# which is the same as where the script is __file__
folder_path = plib.Path(__file__).resolve().parent
# if running as a Jupyter notebook, use absolute paths
# folder_path = plib.Path("absolute path to folder")
# %%
# create the project with default paramters for all samples
proj = Project(
    folder_path,
    temp_unit="K",  # the temperature used in the results (not in the inputs)
    time_moist=38,  # the time (min) where the moisture content shoudl be taken
    time_vm=167,  # the time (min) where the volatile matter should be taken
    dtg_window_filter=51,  # change this to 11, 21, 51, or None to see the difference
)
# add the first sample, giving names and filenames (txt files in the folder path)
sru = Sample(
    project=proj,
    name="sru",
    filenames=["SRU_1", "SRU_2", "SRU_3"],
)
# add the second file (some parameters are overwritten)
misc = Sample(
    project=proj,
    name="misc",
    filenames=["MIS_1", "MIS_2", "MIS_3"],
    time_vm=147,  # change the value for this sample only
)
# for each sample, create tg_dtg plots to visually check results
proximate_samples = [sru, misc]
for sample in proximate_samples:
    _ = sample.plot_tg_dtg()
# create a df report (rep) and export it in excel with all samples
rep = proj.multireport(
    samples=proximate_samples,  # samples to include
    report_type="proximate",  # specify the type of report
    report_style="ave_pm_std",  # specify the style of the report
)

# PLOT CUSTOMIZATION
# plots can be customized using the kwargs in each plotting method
# the list of valid kwargs is in the MyFigure documentation (or code)

# plot the results of the report as a barplot
mf = proj.plot_multireport(
    "proximate_plot",
    samples=proximate_samples,
    report_type="proximate",
    height=4,
    width=4.5,
    y_lim=(0, 100),
    # legend_loc="upper left",
)
# plot the TG curves for each sample as ave+-std
mf = proj.plot_multi_tg(
    "tg_plot",
    samples=proximate_samples,
    height=4,
    width=4.5,
)
# plot the DTG curves for each sample as ave+-std
mf = proj.plot_multi_dtg(
    "dtg_plot",
    samples=proximate_samples,
    height=4,
    width=4.5,
)

# %%
