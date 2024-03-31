# %%
import pathlib as plib
from tga_data_analysis.tga import Project, Sample

test_dir = plib.Path(__file__).resolve().parent / "tga"
test_dir: plib.Path = plib.Path(
    r"C:\Users\mp933\OneDrive - Cornell University\Python\tga_data_analysis\example\tga"
)


# %%
proj_C = Project(test_dir, name="test", temp_unit="C")
proj_K = Project(test_dir, name="test", temp_unit="K")
cell = Sample(
    project=proj_C,
    name="cell",
    filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
    time_moist=38,
    time_vm=None,
)
mf = cell.plot_tg_dtg()
cell = Sample(
    project=proj_K,
    name="cell",
    filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
    time_moist=38,
    time_vm=None,
)
mf = cell.plot_tg_dtg()
# %%

# %%

proj = Project(test_dir, name="test", temp_unit="K")
sru = Sample(
    project=proj, name="misc", filenames=["SRU_1", "SRU_2", "SRU_3"], time_moist=38, time_vm=167
)
misc = Sample(
    project=proj, name="misc", filenames=["MIS_1", "MIS_2", "MIS_3"], time_moist=38, time_vm=147
)
# %%
sda.plot_soliddist()
rep = proj.multireport(report_type="soliddist")
rep = proj.plot_multireport(
    report_type="soliddist",
    legend_loc="upper center",
    color_palette="rocket",
    color_palette_n_colors=7,
)
# %%
dig = Sample(
    project=proj, name="dig", filenames=["DIG10_1", "DIG10_2", "DIG10_3"], time_moist=22, time_vm=98
)
# %%
for sample in proj.samples.values():
    for report_type in [
        "proximate",
        "oxidation",
        "oxidation_extended",
        "soliddist",
        "soliddist_extended",
    ]:
        sample.report(report_type)
    mf = sample.plot_tg_dtg()

# %%
mf = sda.plot_soliddist()

mf = sdb.plot_soliddist()
# %%
misc.deconv_analysis([280 + 273, 380 + 273])
cell.deconv_analysis([310 + 273, 450 + 273, 500 + 273])
mf = misc.plot_deconv()
mf = cell.plot_deconv()
# %%
for report_type in [
    "proximate",
    "oxidation",
    "oxidation_extended",
    "soliddist",
    # "soliddist_extended",  # not supported
]:
    for report_style in ["repl_ave_std", "ave_std", "ave_pm_std"]:
        print(f"{report_type = }, {report_style = }")
        proj.multi_report(report_type=report_type, report_style=report_style)

# %%
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
    filenames=["CLSOx10_2", "CLSOx10_3"],
    time_moist=38,
    time_vm=None,
    heating_rate_deg_min=10,
)
cell_ox50 = Sample(
    project=proj,
    name="cell_ox50",
    load_skiprows=8,
    filenames=["CLSOx50_4", "CLSOx50_5"],
    time_moist=38,
    time_vm=None,
    heating_rate_deg_min=50,
)
# %%
# kas_cell = proj.kas_analysis(samplenames=["cell_ox5", "cell_ox10", "cell_ox50"])
# %%
rep = proj.plot_multireport(
    report_type="proximate",
    x_ticklabels_rotation=30,
    legend_loc="best",
    legend_bbox_xy=(1, 1.01),
)
# %%
rep = proj.plot_multireport(report_type="oxidation", x_ticklabels_rotation=0)
