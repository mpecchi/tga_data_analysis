# %%
import pathlib as plib
from tga_data_analysis.tga import Project, Sample
from tga_data_analysis.kas_kinetics import KasSample, plot_multi_activation_energy

# this is the relative path to the folder where the data is
# which is the same as where the script is __file__
folder_path = plib.Path(__file__).resolve().parent
# if running as a Jupyter notebook, use absolute paths
# folder_path = plib.Path("absolute path to folder")
# %%
# create the project with default paramters for all samples and add samples
proj = Project(
    folder_path=folder_path,
    name="test",
    temp_unit="K",
    time_moist=38,
    time_vm=None,
    load_skiprows=8,
)
cell_ox5 = Sample(
    project=proj,
    name="cell_ox5",
    filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
    heating_rate_deg_min=5,
    load_skiprows=0,
)
cell_ox10 = Sample(
    project=proj,
    name="cell_ox10",
    filenames=["CLSOx10_2", "CLSOx10_3", "CLSOx10_4"],
    heating_rate_deg_min=10,
)
cell_ox50 = Sample(
    project=proj,
    name="cell_ox50",
    filenames=["CLSOx50_4", "CLSOx50_5", "CLSOx50_6"],
    heating_rate_deg_min=50,
)
cell_ox100 = Sample(
    project=proj,
    name="cell_ox100",
    filenames=["CLSOx100_4", "CLSOx100_5", "CLSOx100_6"],
    heating_rate_deg_min=100,
)

pc_ox10 = Sample(
    project=proj,
    name="pc_ox10",
    filenames=["PCOx10_1"],
    heating_rate_deg_min=10,
)
pc_ox50 = Sample(
    project=proj,
    name="pc_ox50",
    filenames=["PCOx50_1"],
    heating_rate_deg_min=50,
)
pc_ox100 = Sample(
    project=proj,
    name="pc_ox100",
    filenames=["PCOx100_1"],
    heating_rate_deg_min=100,
)

# %%
# create a KasSample that includes all the Samples needed for the KAS analysis
# ramps can either be specified for each sample or in the KasSample
cell = KasSample(
    proj,  # specify the project
    samples=[cell_ox5, cell_ox10, cell_ox50, cell_ox100],  # the samples
    name="cell",
)
# create KAS isolines for the cell sample
mf = cell.plot_isolines(legend_bbox_xy=(1, 1))
# plot activation energy for the cell sample
mf = cell.plot_activation_energy(legend_bbox_xy=(1, 1))
# %%
# create a second KasSAmple and do the same operatio
pc = KasSample(proj, samples=[pc_ox10, pc_ox50, pc_ox100], name="pc")
mf = pc.plot_isolines(legend_bbox_xy=(1, 1))
mf = pc.plot_activation_energy(legend_bbox_xy=(1, 1))
# %%
# create a multiplot with activation energies of all specified KasSamples
mf = plot_multi_activation_energy([cell, pc])

# %%
