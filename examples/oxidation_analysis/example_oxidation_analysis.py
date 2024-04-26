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
    folder_path=folder_path,
    temp_unit="K",  # the temperature used in the results (not in the inputs)
    load_skiprows=0,  # number of rows to skip when importing files
    time_moist=38,  # the time (min) where the moisture content shoudl be taken
    time_vm=None,  # specifies that no volatile matter is evaluated
)
# adding samples
cell_ox5 = Sample(
    project=proj,  # the project the sample belongs to
    name="cell_ox5",
    filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
    heating_rate_deg_min=5,  # the heating_rate used in this sample's oxidation
)
cell_ox10 = Sample(
    project=proj,
    name="cell_ox10",
    load_skiprows=8,  # overwriting the default value of 0
    filenames=["CLSOx10_2", "CLSOx10_3", "CLSOx10_4"],
    heating_rate_deg_min=10,
)
# for each single sample create oxidation reports and
# tg and dtg plot for visual check of results
for sample in proj.samples.values():
    for report_type in [
        "oxidation",
        "oxidation_extended",
    ]:
        sample.report(report_type)
    mf = sample.plot_tg_dtg()
# %%
# PLOT CUSTOMIZATION
# plots can be customized using the kwargs in each plotting method
# the list of valid kwargs is in the MyFigure documentation (or code)
# plot the TG curves for each sample as ave+-std
mf = proj.plot_multi_tg()
# plot the DTG curves for each sample as ave+-std
mf = proj.plot_multi_dtg()
# create a df report (rep) and export it in excel with all samples
rep = proj.multireport(report_type="oxidation")
# plot the results of the report as a barplot
mf = proj.plot_multireport(
    report_type="oxidation",
    width=4,
)


# %%
