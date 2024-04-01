import pathlib as plib
from tga_data_analysis.tga import Project, Sample

test_dir = plib.Path(__file__).resolve().parent / "proximate"

proj = Project(test_dir, name="test", temp_unit="K")
sru = Sample(
    project=proj, name="sru", filenames=["SRU_1", "SRU_2", "SRU_3"], time_moist=38, time_vm=167
)
misc = Sample(
    project=proj, name="misc", filenames=["MIS_1", "MIS_2", "MIS_3"], time_moist=38, time_vm=147
)

proximate_samples = [sru, misc]
for sample in proximate_samples:
    _ = sample.plot_tg_dtg()
rep = proj.multireport(
    samples=proximate_samples,
    report_type="proximate",
)
rep = proj.plot_multireport(
    "rep_",
    samples=proximate_samples,
    report_type="proximate",
    height=4,
    width=4.5,
    y_lim=(0, 100),
    # legend_loc="upper left",
)
# %%
rep = proj.plot_multi_dtg(
    "rep",
    samples=proximate_samples,
    height=4,
    width=4.5,
)
