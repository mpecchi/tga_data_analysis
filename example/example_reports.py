# %%
import pathlib as plib
from tga_data_analysis.tga import Project, Sample

test_dir = plib.Path(__file__).resolve().parent / "proximate_analysis"

proj = Project(test_dir, name="test", temp_unit="K")
sru = Sample(
    project=proj, name="sru", filenames=["SRU_1", "SRU_2", "SRU_3"], time_moist=38, time_vm=167
)
misc = Sample(
    project=proj, name="misc", filenames=["MIS_1", "MIS_2", "MIS_3"], time_moist=38, time_vm=147
)

for sample in proj.samples.values():
    for report_type in [
        "proximate",
        "oxidation",
        "oxidation_extended",
        "soliddist",
        "soliddist_extended",
    ]:
        sample.report(report_type)

report_type = "proximate"
for report_style in ["repl_ave_std", "ave_std", "ave_pm_std"]:
    print(f"{report_type = }, {report_style = }")
    _ = proj.multireport(report_type=report_type, report_style=report_style)

_ = proj.plot_multireport(report_type=report_type)

# %%
