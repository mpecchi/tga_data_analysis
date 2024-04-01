# %%
import pytest
import pathlib as plib
import pandas as pd
import numpy as np
from tga_data_analysis.tga import Measure, Sample, Project

# %%
test_dir: plib.Path = plib.Path(
    r"C:\Users\mp933\OneDrive - Cornell University\Python\tga_data_analysis\tests\data"
)
print(test_dir)

# %%
m1 = Measure()
values = [1, 4, 7]
for repl, value in enumerate(values):
    m1.add(repl, value)
assert m1.ave() == np.average(values)
assert m1.std() == np.std(values)
# %%
m2 = Measure()
values = [[1, 4, 5], [2, 6, 7], [3, 8, 9]]
ave = [2, 6, 7]
std = 0
for repl, value in enumerate(values):
    m2.add(repl, value)
print(m2.ave())
print(m2.std())

# %%
p = Project(test_dir)
# %%
mis = Sample(
    name="MIS", filenames=["MIS_1", "MIS_2", "MIS_3"], project=p, folder_path=p.folder_path
)
mis.load_files()
mis.proximate_analysis()
mis.oxidation_analysis()
# %%
mis.temp()
# %%
