# %%
import pathlib as plib
from tga_data_analysis.tga import Project, Sample

test_dir = plib.Path(__file__).resolve().parent / "project"
test_dir: plib.Path = plib.Path(
    r"C:\Users\mp933\OneDrive - Cornell University\Python\tga_data_analysis\example\project"
)


# %% test different units
proj_default = Project(
    test_dir,
    name="default",
    temp_unit="C",
    plot_font="Dejavu Sans",
    dtg_basis="temperature",
    resolution_sec_deg_dtg=5,
    dtg_window_filter=101,
    plot_grid=False,
    temp_initial_celsius=40,
    temp_lim_dtg_celsius=None,
)
cell = Sample(
    project=proj_default,
    name="cell",
    filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
    time_moist=38,
    time_vm=None,
)
mf = cell.plot_tg_dtg()
proj_mod = Project(
    test_dir,
    name="mod",
    temp_unit="K",
    plot_font="Times New Roman",
    dtg_basis="time",
    resolution_sec_deg_dtg=2,  # bad choice, but shows the parameter use (see DTG (db))
    dtg_window_filter=201,  # bad choice, but shows the parameter use (see DTG (db))
    plot_grid=True,
    temp_initial_celsius=50,
    temp_lim_dtg_celsius=(150, 750),
)
cell = Sample(
    project=proj_mod,
    name="cell",
    filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
    time_moist=38,
    time_vm=None,
)
mf = cell.plot_tg_dtg()
