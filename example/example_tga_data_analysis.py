# example of hplc_data_analysis
# %%
import tga_data_analysis as tga

# %%
# this has to be changed to where the _example/data folder is
folder_path = r"C:\Users\mp933\OneDrive - Cornell University\Python\tga_data_analysis\example\data"
# hange this to folder_path = plib.Path(r"C:\Path\To\Your\Data") for your project
# class methods need to be called at the beginning to influence all instances
tga.TGAExp.set_folder_path(folder_path)
tga.TGAExp.set_plot_grid(True)
tga.TGAExp.set_plot_font("Times New Roman")
tga.TGAExp.set_T_unit("Kelvin")
# %%
# objects are created with the new default values
P1 = tga.TGAExp(name="P1", filenames=["MIS_1", "MIS_2", "MIS_3"], time_moist=38, time_vm=147)
P2 = tga.TGAExp(
    name="P2",
    load_skiprows=0,
    filenames=["DIG10_1", "DIG10_2", "DIG10_3"],
    time_moist=22,
    time_vm=98,
)
# %%
Ox5 = tga.TGAExp(
    name="Ox5", filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"], time_moist=38, time_vm=None
)
Ox10 = tga.TGAExp(
    name="Ox10", load_skiprows=8, filenames=["CLSOx10_2", "CLSOx10_3"], time_moist=38, time_vm=None
)
Ox50 = tga.TGAExp(
    name="Ox50", load_skiprows=8, filenames=["CLSOx50_4", "CLSOx50_5"], time_moist=38, time_vm=None
)
SD1 = tga.TGAExp(name="SDa", filenames=["SDa_1", "SDa_2", "SDa_3"], time_moist=38, time_vm=None)
SD2 = tga.TGAExp(name="SDb", filenames=["SDb_1", "SDb_2", "SDb_3"], time_moist=38, time_vm=None)

# %%
a = P1.proximate_report()
b = P2.proximate_report()
c = Ox5.oxidation_report()
d = Ox10.oxidation_report()
e = Ox50.oxidation_report()
f = SD1.soliddist_report()
g = SD2.soliddist_report()
# %%
mf = P1.tg_plot_new(filename="a")

# %%
P1.tg_plot()
# %%

P1.deconv_analysis([280, 380])
Ox5.deconv_analysis([310, 450, 500])
# %%
tg_multi_plot([P1, P2, Ox5, SD1], filename="P1P2Ox5SD1")
dtg_multi_plot([P1, P2, Ox5, SD1], filename="P1P2Ox5SD1")
h = proximate_multi_report([P1, P2, Ox5, SD1], filename="P1P2Ox5SD1")
proximate_multi_plot([P1, P2, Ox5, SD1], filename="P1P2Ox5SD1", bboxtoanchor=False)
i = oxidation_multi_report([Ox5, Ox10, Ox50], filename="Ox5Ox10Ox50")
oxidation_multi_plot([Ox5, Ox10, Ox50], filename="Ox5Ox10Ox50")  # yLim=[250, 400],
j = soliddist_multi_report([SD1, SD2], filename="SD1SD2")
soliddist_multi_plot([SD1, SD2], filename="SD1SD2")

# %%
k = KAS_analysis([Ox5, Ox10, Ox50], [5, 10, 50])
KAS_plot_isolines([Ox5], filename="Ox5Ox10Ox50")
KAS_plot_Ea([Ox5, Ox5], filename="Ox5Ox10Ox50", bboxtoanchor=False, leg_cols=2)
