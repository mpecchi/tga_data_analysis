# %%
import pathlib as plib
from tga_data_analysis.tga import Project, Sample

# this is the relative path to the folder where the data is
# which is the same as where the script is __file__
folder_path = plib.Path(__file__).resolve().parent
# if running as a Jupyter notebook, use absolute paths
# folder_path = plib.Path("absolute path to folder")
folder_path = plib.Path("/Users/matteo/Projects/tga_data_analysis/examples/reports")
# %%
# create the project with default paramters for all samples
proj = Project(
    folder_path,
    temp_unit="K",  # the temperature used in the results (not in the inputs)
    time_moist=38,  # the time (min) where the moisture content shoudl be taken
    time_vm=167,  # the time (min) where the volatile matter should be taken
    dtg_basis="temperature",  # either "time" or "temperature" (should be temperature)
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

# create all available reports for all samples
# a report can contain useless data if the sample is not designed for that analysis
# example the proximate report for an oxidation run is garbage
# this part explains how to run each report, but the user should
# select the ones that make sense on the specific sample
for sample in proj.samples.values():
    for report_type in [
        "proximate",
        "oxidation",
        "oxidation_extended",
        "soliddist",
        "soliddist_extended",
    ]:
        sample.report(report_type)

# each report can be produced according to different styles
# examples are:
# repl_ave_std: for each replicate the ave and std are given as seprate values
# ave_std: ave and std are given as seprated values (no replicate info)
# ave_pm_std: results are in the form ave+-std
# the following examples use the proximate type
report_type = "proximate"
for report_style in ["repl_ave_std", "ave_std", "ave_pm_std"]:
    print(f"{report_type = }, {report_style = }")
    _ = proj.multireport(report_type=report_type, report_style=report_style)

# PLOT CUSTOMIZATION
# plots can be customized using the kwargs in each plotting method
# the list of valid kwargs is in the MyFigure documentation (or code)

# plot the tabulated results as a barplot
mf = proj.plot_multireport(report_type=report_type)
