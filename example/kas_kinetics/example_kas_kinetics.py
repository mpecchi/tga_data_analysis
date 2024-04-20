import pathlib as plib
from tga_data_analysis.tga import Project, Sample
from tga_data_analysis.kas_kinetics import KasSample, plot_multi_activation_energy


folder_path = plib.Path(__file__).resolve().parent
# %%
proj = Project(folder_path=folder_path, name="test", temp_unit="K")
cell_ox5 = Sample(
    project=proj,
    name="cell_ox5",
    filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
    time_moist=38,
    time_vm=None,
    heating_rate_deg_min=5,
)
cell_ox10 = Sample(
    project=proj,
    name="cell_ox10",
    load_skiprows=8,
    filenames=["CLSOx10_2", "CLSOx10_3", "CLSOx10_4"],
    time_moist=38,
    time_vm=None,
    heating_rate_deg_min=10,
)
cell_ox50 = Sample(
    project=proj,
    name="cell_ox50",
    load_skiprows=8,
    filenames=["CLSOx50_4", "CLSOx50_5", "CLSOx50_6"],
    time_moist=38,
    time_vm=None,
    heating_rate_deg_min=50,
)
cell_ox100 = Sample(
    project=proj,
    name="cell_ox100",
    load_skiprows=8,
    filenames=["CLSOx100_4", "CLSOx100_5", "CLSOx100_6"],
    time_moist=38,
    time_vm=None,
    heating_rate_deg_min=100,
)
# %%
pc_ox10 = Sample(
    project=proj,
    name="pc_ox10",
    load_skiprows=8,
    filenames=["PCOx10_1"],
    time_moist=38,
    time_vm=None,
    heating_rate_deg_min=10,
)
pc_ox50 = Sample(
    project=proj,
    name="pc_ox50",
    load_skiprows=8,
    filenames=["PCOx50_1"],
    time_moist=38,
    time_vm=None,
    heating_rate_deg_min=50,
)
pc_ox100 = Sample(
    project=proj,
    name="pc_ox100",
    load_skiprows=8,
    filenames=["PCOx100_1"],
    time_moist=38,
    time_vm=None,
    heating_rate_deg_min=100,
)

# %%
cell = KasSample(proj, samples=[cell_ox5, cell_ox10, cell_ox50, cell_ox100], name="cellulose")
mf = cell.plot_isolines(legend_bbox_xy=(1, 1))
mf = cell.plot_activation_energy(legend_bbox_xy=(1, 1))
# %%
pc = KasSample(proj, samples=[pc_ox10, pc_ox50, pc_ox100], name="primary")
mf = pc.plot_isolines(legend_bbox_xy=(1, 1))
mf = pc.plot_activation_energy(legend_bbox_xy=(1, 1))
# %%
mf = plot_multi_activation_energy([cell, pc])
