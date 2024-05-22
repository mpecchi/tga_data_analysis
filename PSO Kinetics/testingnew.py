#%%
import pathlib as plib
from tga_data_analysis.tga import Project, Sample

# this is the relative path to the folder where the data is
# which is the same as where the script is __file__
folder_path = plib.Path('/Users/charlottealbunio/Documents/Python/tga_data_analysis/PSO Kinetics/data')
# if running as a Jupyter notebook, use absolute paths
# folder_path = r"absolute path to folder"
# %%
# create the project instance, important parameters are:
# the moisture value is used to compute the dry basis curve

proj_default = Project(
    folder_path,
    name="sawdust",  # the name of the project
    temp_unit="C",  # the temperature that results will use (C or K)
    plot_font="Dejavu Sans",  # chose the font for the plots
    resolution_sec_deg_dtg=5,  # chose the resolution for dtg vectors
    dtg_window_filter=None,  # chose the filtering window for dtg curve
    plot_grid=False,  # wheter to include a grid in plots
    temp_initial_celsius=40,  # initial temperature for all curves (exclude data before)
    temp_lim_dtg_celsius=(120,615),  # temperature limits for the dtg curves
    time_moist=38,  # the time where mass loss due to moisture is computed,
    time_vm=None, 
    load_skiprows=8,

)

sw5 = Sample(
    project=proj_default,
    name="softwood 5K",
    filenames=["softwood-5Kmin_1", "softwood-5Kmin_2", "softwood-5Kmin_3"],
    time_moist=10,
    time_vm=None,
    load_skiprows=8,
)

sw10 = Sample(
    project=proj_default,
    name="softwood 10K",
    filenames=["softwood-10Kmin_1", "softwood-10Kmin_2", "softwood-10Kmin_3"],
    time_moist=10,
    time_vm=None,
    load_skiprows=8,
    
)

sw20 = Sample(
    project=proj_default,
    name="softwood 20K",
    filenames=["softwood-20Kmin_1", "softwood-20Kmin_2", "softwood-20Kmin_3"],
    time_moist=10,
    time_vm=None,
    load_skiprows=8,
)

sw50 = Sample(
    project=proj_default,
    name="softwood 50K",
    filenames=["softwood-50Kmin_1", "softwood-50Kmin_2", "softwood-50Kmin_3"],
    time_moist=10,
    time_vm=None,
    load_skiprows=8,
)
sw50.dtg_window_filter=50

mf = sw50.plot_tg_dtg()
sw50.ddtg_analysis()
#%% Plotting and finding peaks for ddtg THIS WORKS PRETTY WELL JUST NEED TO FIND MINIMA
from scipy import signal
sw50.dtg_window_filter=50
sw50.ddtg_analysis()
inv_data=abs(sw50.ddtg_db.ave())
min_peakind, _=signal.find_peaks(inv_data)
print(min_peakind)
# %%
#plt.plot(sw5.temp_dtg()[min_peakind],abs(sw5.ddtg_db()[min_peakind]),"x")
plt.plot(sw50.temp_dtg(),abs(sw50.ddtg_db()))
plt.plot(sw50.temp_dtg(), abs(sw50.dtg_db()))
# %%
import pandas as pd 

df = pd.DataFrame({"50 K/min Temp" : sw50.temp_dtg(), "DTG_50": sw50.dtg_db(), "DDTG_50": abs(sw50.ddtg_db())})
df.to_csv("/Users/charlottealbunio/Downloads/50_k_min.csv", index="false")
# %%
