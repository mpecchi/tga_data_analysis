import pathlib as plib
from tga_data_analysis.tga import Project, Sample

folder_path = plib.Path(__file__).resolve().parent / "oxidation_analysis"


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

for sample in proj.samples.values():
    for report_type in [
        "oxidation",
        "oxidation_extended",
    ]:
        sample.report(report_type)
    mf = sample.plot_tg_dtg()


rep = proj.multireport(report_type="oxidation")
mf = proj.plot_multi_dtg()
