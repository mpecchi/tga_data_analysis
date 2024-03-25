# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:28:04 2022

@author: mp933
"""
# %%
import pathlib as plib
import numpy as np
import pandas as pd
from typing import Any
from scipy.signal import savgol_filter
from lmfit.models import GaussianModel, LinearModel

from myfigure import MyFigure  # !!!!


class Measure:
    """
    A class to collect and analyze a series of numerical data points or arrays.

    Attributes:
        _stk (dict): A dictionary to store the data points or numpy arrays with integer keys.
    """

    def __init__(self):
        """
        Initializes the Measure class with an empty dictionary for _stk.
        """
        self._stk: dict[int : np.ndarray | float] = {}
        self._ave: np.ndarray | float | None = None
        self._std: np.ndarray | float | None = None

    def add(self, replicate: int, value: np.ndarray | pd.Series | float | int) -> None:
        """
        Adds a new data point or numpy array to the _stk dictionary with the specified replicate key.

        :param replicate: The integer key (replicate number) to associate with the value.
        :param value: A data point, numpy array, or pandas Series to be added.
        """
        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.to_numpy()
        elif isinstance(value, np.ndarray):
            value = value.flatten()
        self._stk[replicate] = value

    def stk(self, replicate: int | None = None) -> np.ndarray | float:
        """
        Retrieves the data point or array associated with the specified replicate key.

        :param replicate: The key (replicate number) of the data point or array to retrieve.
        :return: The data point or array associated with the replicate key.
        """
        if replicate is None:
            return self._stk
        else:
            return self._stk.get(replicate)

    def ave(self) -> np.ndarray:
        """
        Calculates the mean of all data points or arrays stored in _stk.

        :return: The mean value(s) across all data points or arrays.
        """
        if all(isinstance(v, np.ndarray) for v in self._stk.values()):
            self._ave = np.mean(np.column_stack(list(self._stk.values())), axis=1)
            return self._ave
        else:
            self._ave = np.mean(list(self._stk.values()))
            return self._ave

    def std(self) -> np.ndarray:
        """
        Calculates the standard deviation of all data points or arrays stored in _stk.

        :return: The standard deviation value(s) across all data points or arrays.
        """
        if all(isinstance(v, np.ndarray) for v in self._stk.values()):
            self._std = np.std(np.column_stack(list(self._stk.values())), axis=1)
            return self._std
        else:
            self._std = np.std(list(self._stk.values()))
            return self._std


class Sample:

    def __init__(
        self,
        name: str,
        filenames: list[str],
        folder_path: plib.Path,
        column_name_mapping: dict[str:str] | None = None,
        load_skiprows: int = 0,
        label: str | None = None,
        time_moist: float = 38.0,
        time_vm: float = 147,
        temp_unit: str = "Celsius",
        temp_initial_celsius: float = 40,
        temp_lim_dtg_celsius: tuple[float] | None = None,
        correct_ash_mg: list[float] | None = None,
        correct_ash_fr: list[float] | None = None,
        resolution_sec_deg_dtg: float = 5.0,
        dtg_basis: str = "temperature",
        dtg_w_savgol_filter: int = 101,
    ):

        if column_name_mapping is None:
            column_name_mapping = {
                "Time": "t_min",
                "Temperature": "T_C",
                "Weight": "m_p",
                "Weight.1": "m_mg",
                "Heat Flow": "heatflow_mW",
                "##Temp./>C": "T_C",
                "Time/min": "t_min",
                "Mass/%": "m_p",
                "Segment": "segment",
            }

        if filenames is None:
            self.filenames = []
        else:
            self.filenames = filenames
        if not label:
            self.label = name
        else:
            self.label = label
        self.folder_path = folder_path
        self.column_name_mapping = column_name_mapping
        self.name = name
        self.n_repl = len(self.filenames)
        self.load_skiprows = load_skiprows

        self.time_moist = time_moist
        self.time_vm = time_vm
        self.correct_ash_mg = correct_ash_mg
        self.correct_ash_fr = correct_ash_fr
        self.temp_unit = temp_unit
        self.dtg_basis = dtg_basis
        self.temp_initial_celsius = temp_initial_celsius
        self.dtg_w_savgol_filter = dtg_w_savgol_filter
        self.resolution_sec_deg_dtg = resolution_sec_deg_dtg

        if temp_lim_dtg_celsius is None:
            self.temp_lim_dtg_celsius = [120, 880]
        else:
            self.temp_lim_dtg_celsius = temp_lim_dtg_celsius
        if self.temp_unit == "Celsius":
            self.temp_lim_dtg = temp_lim_dtg_celsius
        elif self.temp_unit == "Kelvin":
            self.temp_lim_dtg = [t + 273.15 for t in self.temp_lim_dtg_celsius]
        else:
            raise ValueError(f"{self.temp_unit = } is not acceptable")

        print(f"{self.temp_lim_dtg=}, {temp_lim_dtg_celsius=}, {self.resolution_sec_deg_dtg=}")
        self.len_dtg_db: int = int(
            (self.temp_lim_dtg[1] - self.temp_lim_dtg[0]) * self.resolution_sec_deg_dtg
        )
        self.temp_dtg: np.ndarray = np.linspace(
            self.temp_lim_dtg[0], self.temp_lim_dtg[1], self.len_dtg_db
        )
        # for variables and computations
        self.files: dict[str : pd.DataFrame] = {}
        self.len_files: dict[str : pd.DataFrame] = {}
        self.len_sample: int = 0

        #
        self.temp: "Measure" = Measure()
        self.time: "Measure" = Measure()
        self.m_ar: "Measure" = Measure()
        self.mp_ar: "Measure" = Measure()
        self.idx_moist: "Measure" = Measure()
        self.idx_vm: "Measure" = Measure()
        self.moist_ar: "Measure" = Measure()
        self.ash_ar: "Measure" = Measure()
        self.fc_ar: "Measure" = Measure()
        self.vm_ar: "Measure" = Measure()
        self.mp_db: "Measure" = Measure()
        self.ash_db: "Measure" = Measure()
        self.fc_db: "Measure" = Measure()
        self.vm_db: "Measure" = Measure()
        self.mp_daf: "Measure" = Measure()
        self.fc_daf: "Measure" = Measure()
        self.vm_daf: "Measure" = Measure()
        self.time_dtg: "Measure" = Measure()
        self.mp_db_dtg: "Measure" = Measure()
        self.dtg_db: "Measure" = Measure()
        self.ave_dev_tga_perc: float | None = None
        # Flag to track if data is loaded
        self.proximate_computed = False
        self.files_loaded = False
        self.oxidation_computed = False
        self.soliddist_computed = False
        self.deconv_computed = False
        self.KAS_computed = False
        # for reports
        self.proximate_report_computed = False
        self.oxidation_report_computed = False
        self.soliddist_report_computed = False
        self.deconv_report_computed = False
        self.KAS_report_computed = False

    def load_files(self):
        """
        Loads all the files for the experiment.

        Returns:
            list: The list of loaded files.
        """
        print("\n" + self.name)
        # import files and makes sure that replicates have the same size
        for filename in self.filenames:
            print(filename)
            # file = self.load_single_file(filename)[0]
            path = plib.Path(self.folder_path, filename + ".txt")
            if not path.is_file():
                path = plib.Path(self.folder_path, filename + ".csv")
            file = pd.read_csv(path, sep="\t", skiprows=self.load_skiprows)
            if file.shape[1] < 3:
                file = pd.read_csv(path, sep=",", skiprows=self.load_skiprows)

            file = file.rename(
                columns={col: self.column_name_mapping.get(col, col) for col in file.columns}
            )
            for column in file.columns:
                file[column] = pd.to_numeric(file[column], errors="coerce")
            file.dropna(inplace=True)
            self.data_loaded = True  # Flag to track if data is loaded
            # FILE CORRECTION
            if self.correct_ash_mg is not None:
                file["m_mg"] = file["m_mg"] - np.min(file["m_mg"]) + self.correct_ash_mg
            try:
                if file["m_mg"].iloc[-1] < 0:
                    print(
                        "neg. mass correction: Max [mg]",
                        np.round(np.max(file["m_mg"]), 3),
                        "; Min [mg]",
                        np.round(np.min(file["m_mg"]), 3),
                    )
                    file["m_mg"] = file["m_mg"] - np.min(file["m_mg"])
            except KeyError:
                file["m_mg"] = file["m_p"]
            file["m_p"] = file["m_mg"] / np.max(file["m_mg"]) * 100
            if self.correct_ash_fr is not None:
                file["m_p"] = file["m_p"] - np.min(file["m_p"]) + self.correct_ash_fr
                file["m_p"] = file["m_p"] / np.max(file["m_p"]) * 100
            file = file[file["T_C"] >= self.temp_initial_celsius].copy()
            file["T_K"] = file["T_C"] + 273.15
            self.files[filename] = file
            self.len_files[filename] = max(file.shape)
        self.len_sample = min(self.len_files.values())
        # keep the shortest vector size for all replicates, create the object
        for fn, fl in self.files.items():
            self.files[fn] = self.files[fn].head(self.len_sample)
        self.files_loaded = True  # Flag to track if data is loaded
        return self.files

    def proximate_analysis(self):
        """
        Performs proximate analysis on the loaded data.
        """
        if not self.data_loaded:
            self.load_files()

        for f, file in enumerate(self.files.values()):
            if self.temp_unit == "Celsius":
                self.temp.add(f, file["T_C"])
            elif self.temp_unit == "Kelvin":
                self.temp.add(f, file["T_K"])
            self.time.add(f, file["t_min"])

            self.m_ar.add(f, file["m_mg"])
            self.mp_ar.add(f, file["m_p"])

            self.idx_moist.add(f, np.argmax(self.time.stk(f) > self.time_moist + 0.01))

            self.moist_ar.add(f, 100 - self.mp_ar.stk(f)[self.idx_moist.stk(f)])
            self.ash_ar.add(f, self.mp_ar.stk(f)[-1])
            self.mp_db.add(f, self.mp_ar.stk(f) * 100 / (100 - self.moist_ar.stk(f)))
            self.mp_db.add(f, np.where(self.mp_db.stk(f) > 100, 100, self.mp_db.stk(f)))
            self.ash_db.add(f, self.ash_ar.stk(f) * 100 / (100 - self.moist_ar.stk(f)))
            self.mp_daf.add(
                f, ((self.mp_db.stk(f) - self.ash_db.stk(f)) * 100 / (100 - self.ash_db.stk(f)))
            )
            if self.time_vm is not None:
                self.idx_vm.add(f, np.argmax(self.time.stk(f) > self.time_vm))
                self.fc_ar.add(f, self.mp_ar.stk(f)[self.idx_vm.stk(f)] - self.ash_ar.stk(f))
                self.vm_ar.add(
                    f, (100 - self.moist_ar.stk(f) - self.ash_ar.stk(f) - self.fc_ar.stk(f))
                )
                self.vm_db.add(f, self.vm_ar.stk(f) * 100 / (100 - self.moist_ar.stk(f)))
                self.fc_db.add(f, self.fc_ar.stk(f) * 100 / (100 - self.moist_ar.stk(f)))

                self.vm_daf.add(
                    f, ((self.vm_db.stk(f) - self.ash_db.stk(f)) * 100 / (100 - self.ash_db.stk(f)))
                )
                self.fc_daf.add(
                    f, ((self.fc_db.stk(f) - self.ash_db.stk(f)) * 100 / (100 - self.ash_db.stk(f)))
                )

            idxs_dtg = [
                np.argmax(self.temp.stk(f) > self.temp_lim_dtg[0]),
                np.argmax(self.temp.stk(f) > self.temp_lim_dtg[1]),
            ]
            # temp_dtg is taken fixed

            # time start from 0 and consideres a fixed heating rate
            self.time_dtg.add(
                f,
                np.linspace(
                    0,
                    self.time.stk(f)[idxs_dtg[1]] - self.time.stk(f)[idxs_dtg[0]],
                    self.len_dtg_db,
                ),
            )

            self.mp_db_dtg.add(
                f,
                np.interp(
                    self.temp_dtg,
                    self.temp.stk(f)[idxs_dtg[0] : idxs_dtg[1]],
                    self.mp_db.stk(f)[idxs_dtg[0] : idxs_dtg[1]],
                ),
            )
            # the combusiton indexes use rates as /min
            if self.dtg_basis == "temperature":
                dtg = np.gradient(self.mp_db_dtg.stk(f), self.temp_dtg)
            if self.dtg_basis == "time":
                dtg = np.gradient(self.mp_db_dtg.stk(f), self.time_dtg.stk(f))
            self.dtg_db.add(f, savgol_filter(dtg, self.dtg_w_savgol_filter, 1))
        # average
        self.ave_dev_tga_perc = np.average(self.mp_db_dtg.std())
        print(
            f"Average TG [%] St. Dev. for replicates: {self.mp_db_dtg.std():0.2f}"
            + str(round(np.average(), 2))
            + " %"
        )
        self.proximate_computed = True

    def oxidation_analysis(self):
        """
        Perform oxidation analysis on the data.

        This method calculates various oxidation parameters based on the proximate analysis results.
        It computes the Ti, Tp, Tb, dwdT_max, dwdT_mean, and S values for each file in the dataset,
        and then calculates the average and standard deviation of these values.

        Returns:
            None
        """
        if not self.proximate_computed:
            self.proximate_analysis()
        self.Ti_idx_stk = np.zeros(self.n_repl, dtype=int)
        self.Ti_stk = np.zeros(self.n_repl)
        self.Tp_idx_stk = np.zeros(self.n_repl, dtype=int)
        self.Tp_stk = np.zeros(self.n_repl)
        self.Tb_idx_stk = np.zeros(self.n_repl, dtype=int)
        self.Tb_stk = np.zeros(self.n_repl)
        self.dwdT_max_stk = np.zeros(self.n_repl)
        self.dwdT_mean_stk = np.zeros(self.n_repl)
        self.S_stk = np.zeros(self.n_repl)
        for f, file in enumerate(self.files):
            threshold = np.max(np.abs(self.dtg_db.stk(f))) * TGAExp.TiTb_threshold
            # Ti = T at which dtg > Ti_thresh wt%/min after moisture removal
            self.Ti_idx_stk[f] = int(np.argmax(np.abs(self.dtg_db.stk(f)) > threshold))
            self.Ti_stk[f] = self.T_dtg[self.Ti_idx_stk[f]]
            # Tp is the T of max abs(dtg)
            self.Tp_idx_stk[f] = int(np.argmax(np.abs(self.dtg_db.stk(f))))
            self.Tp_stk[f] = self.T_dtg[self.Tp_idx_stk[f]]
            # Tb reaches < 1 wt%/min at end of curve
            try:
                self.Tb_idx_stk[f] = int(np.flatnonzero(self.dtg_db.stk(f) < -threshold)[-1])
            except IndexError:  # the curve nevers goes above 1%
                self.Tb_idx_stk[f] = 0
            self.Tb_stk[f] = self.T_dtg[self.Tb_idx_stk[f]]

            self.dwdT_max_stk[f] = np.max(np.abs(self.dtg_db.stk(f)))
            self.dwdT_mean_stk[f] = np.average(np.abs(self.dtg_db.stk(f)))
            # combustion index
            self.S_stk[f] = (
                self.dwdT_max_stk[f]
                * self.dwdT_mean_stk[f]
                / self.Ti_stk[f]
                / self.Ti_stk[f]
                / self.Tb_stk[f]
            )
        # # average
        self.Ti = np.average(self.Ti_stk)
        self.Ti_std = np.std(self.Ti_stk)
        self.Tb = np.average(self.Tb_stk)
        self.Tb_std = np.std(self.Tb_stk)

        self.dwdT_max = np.average(self.dwdT_max_stk)
        self.dwdT_max_std = np.std(self.dwdT_max_stk)
        self.Tp = np.average(self.Tp_stk)
        self.Tp_std = np.std(self.Tp_stk)
        self.dwdT_mean = np.average(self.dwdT_mean_stk)
        self.dwdT_mean_std = np.std(self.dwdT_mean_stk)
        self.S = np.average(self.S_stk)
        self.S_std = np.std(self.S_stk)
        self.oxidation_computed = True


# %%
if __name__ == "__main__":

    a = Measure()
    a.add(1, np.array([1, 2, 3]))
    a.add(2, np.array([2, 3, 4]))
    a.add(3, np.array([3, 4, 5]))
    assert np.allclose(a.ave(), np.array([2, 3, 4]))
    assert np.allclose(a.std(), np.array([0.81649658, 0.81649658, 0.81649658]))

    path: plib.Path = plib.Path(
        r"C:\Users\mp933\OneDrive - Cornell University\Python\tga_data_analysis\example\data"
    )
    mis: "Sample" = Sample(
        folder_path=path,
        name="P1",
        filenames=["MIS_1", "MIS_2", "MIS_3"],
        time_moist=38,
        time_vm=147,
    )
    mis.load_files()
    m1: pd.DataFrame = mis.files["MIS_1"]
    m1["T_C"]
    # %%
    mis.proximate_analysis()
    # def test_Measure()
# %%