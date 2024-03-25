# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:28:04 2022

@author: mp933
"""
import pathlib as plib
import numpy as np
import pandas as pd
from typing import Any
from scipy.signal import savgol_filter as SavFil
from lmfit.models import GaussianModel, LinearModel

from .plotting import lnstls, clrs, figure_create, figure_save
from .myfigure import MyFigure


class TGAExp:
    """
    Represents a TGA (Thermogravimetric Analysis) Experiment.

    Attributes:
        folder (str): The folder name for the experiment.
        in_path (str): The input path for the experiment.
        out_path (str): The output path for the experiment.
        T_unit (str): The unit of temperature used in the experiment.
        T_symbol (str): The symbol for the temperature unit.
        plot_font (str): The font used for plotting.
        plot_grid (bool): Flag indicating whether to show grid in plots.
        dtg_basis (str): The basis for calculating the derivative thermogravimetric (DTG) curve.
        resolution_T_dtg (int): The resolution of the temperature range for the DTG curve.
        dtg_w_SavFil (int): The window size for Savitzky-Golay filtering of the DTG curve.

    Methods:
        __init__(... (see below)):
            Initializes a TGAExp object with the specified parameters.

        load_single_file(self, filename):
            Loads a single file for the experiment.

        load_files(self):
            Loads all the files for the experiment.

        proximate_analysis(self):
            Performs proximate analysis on the loaded data.
    """

    folder_path = plib.Path.cwd()
    in_path = folder_path
    out_path = plib.Path(in_path, "Output")
    T_unit = "Celsius"
    if T_unit == "Celsius":
        T_symbol = "°C"
    elif T_unit == "Kelvin":
        T_symbol = "K"
    plot_font = "Dejavu Sans"
    plot_grid = False
    dtg_basis = "temperature"
    if dtg_basis == "temperature":
        DTG_lab = "DTG [wt%/" + T_symbol + "]"
    elif dtg_basis == "time":
        DTG_lab = "DTG [wt%/min]"
    TiTb_threshold = 0.01  # % of the peak that is used for Ti and Tb
    resolution_T_dtg = 5
    dtg_w_SavFil = 101
    TG_lab = "TG [wt%]"
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

    @classmethod
    def set_folder_path(cls, path):
        """necessary to specify the folder path with files"""
        cls.folder_path = plib.Path(path).resolve()
        cls.in_path = cls.folder_path
        cls.out_path = plib.Path(cls.in_path, "Output")
        plib.Path(cls.out_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def set_T_unit(cls, new_T_unit):
        # Update paths based on new folder
        cls.T_unit = new_T_unit
        if cls.T_unit == "Celsius":
            cls.T_symbol = "°C"
        elif cls.T_unit == "Kelvin":
            cls.T_symbol = "K"

    @classmethod
    def set_plot_font(cls, new_plot_font):
        cls.plot_font = new_plot_font

    @classmethod
    def set_plot_grid(cls, new_plot_grid):
        cls.plot_grid = new_plot_grid

    @classmethod
    def set_dtg_basis(cls, new_dtg_basis):
        cls.dtg_basis = new_dtg_basis
        if new_dtg_basis == "temperature":
            cls.DTG_lab = "DTG [wt%/" + cls.T_symbol + "]"
        elif new_dtg_basis == "time":
            cls.DTG_lab = "DTG [wt%/min]"

    @classmethod
    def set_resolution_T_dtg(cls, new_resolution_T_dtg):
        cls.resolution_T_dtg = new_resolution_T_dtg

    @classmethod
    def set_dtg_w_SavFil(cls, new_dtg_w_SavFil):
        cls.dtg_w_SavFil = new_dtg_w_SavFil

    def __init__(
        self,
        name,
        filenames,
        load_skiprows=0,
        label=None,
        time_moist=38,
        time_vm=147,
        T_initial_C=40,
        Tlims_dtg_C=[120, 880],
        correct_ash_mg=None,
        correct_ash_fr=None,
        oxid_Tb_thresh=None,
    ):
        """
        Initializes a TGAExp object with the specified parameters.

        Args:
            name (str): The name of the experiment.
            filenames (list): The list of filenames for the experiment.
            load_skiprows (int, optional): The number of rows to skip while loading the files. Defaults to 0.
            label (str, optional): The label for the experiment. Defaults to None.
            time_moist (int, optional): The time for moisture analysis. Defaults to 38.
            time_vm (int, optional): The time for volatile matter analysis. Defaults to 147.
            T_initial_C (int, optional): The initial temperature in Celsius. Defaults to 40.
            Tlims_dtg_C (list, optional): The temperature limits for the DTG curve in Celsius. Defaults to [120, 800].
            correct_ash_mg (float, optional): The correction value for ash mass in mg. Defaults to None.
            correct_ash_fr (float, optional): The correction value for ash fraction. Defaults to None.
            oxid_Tb_thresh (int, optional): The threshold for oxidation temperature. Defaults to 1.
        """
        self.name = name
        if filenames is None:
            self.filenames = []
        else:
            self.filenames = filenames
        if not label:
            self.label = name
        else:
            self.label = label
        self.n_repl = len(self.filenames)
        self.load_skiprows = load_skiprows

        self.time_moist = time_moist
        self.time_vm = time_vm
        self.correct_ash_mg = correct_ash_mg
        self.correct_ash_fr = correct_ash_fr
        self.T_initial_C = T_initial_C
        if TGAExp.T_unit == "Celsius":
            self.Tlims_dtg = Tlims_dtg_C
        elif TGAExp.T_unit == "Kelvin":
            self.Tlims_dtg = [T + 273.15 for T in Tlims_dtg_C]
        # for variables and computations
        self.data_loaded = False  # Flag to track if data is loaded
        self.proximate_computed = False
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

        self.create_tgaexp()

    def create_tgaexp(self) -> "TGAExp":
        return self

    def load_single_file(self, filename):
        """
        Loads a single file for the experiment.

        Args:
            filename (str): The filename of the file to load.

        Returns:
            list: The list of loaded files.
        """
        path = plib.Path(TGAExp.in_path, filename + ".txt")
        if not path.is_file():
            path = plib.Path(TGAExp.in_path, filename + ".csv")
        file = pd.read_csv(path, sep="\t", skiprows=self.load_skiprows)
        if file.shape[1] < 3:
            file = pd.read_csv(path, sep=",", skiprows=self.load_skiprows)

        file = file.rename(
            columns={col: TGAExp.column_name_mapping.get(col, col) for col in file.columns}
        )
        for column in file.columns:
            file[column] = pd.to_numeric(file[column], errors="coerce")
        file.dropna(inplace=True)
        self.files = [file]
        self.data_loaded = True  # Flag to track if data is loaded
        return self.files

    def load_files(self):
        """
        Loads all the files for the experiment.

        Returns:
            list: The list of loaded files.
        """
        print("\n" + self.name)
        # import files and makes sure that replicates have the same size
        (
            files,
            len_files,
        ) = (
            [],
            [],
        )
        for filename in self.filenames:
            print(filename)
            file = self.load_single_file(filename)[0]
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
            file = file[file["T_C"] >= self.T_initial_C].copy()
            file["T_K"] = file["T_C"] + 273.15
            files.append(file)
            len_files.append(max(file.shape))
        self.len_sample = np.min(len_files)
        # keep the shortest vector size for all replicates, create the object
        self.files = [f.head(self.len_sample) for f in files]
        self.data_loaded = True  # Flag to track if data is loaded
        return self.files

    def proximate_analysis(self):
        """
        Performs proximate analysis on the loaded data.
        """
        if not self.data_loaded:
            self.load_files()

        self.T_stk = np.zeros((self.len_sample, self.n_repl))
        self.time_stk = np.zeros((self.len_sample, self.n_repl))
        self.m_ar_stk = np.zeros((self.len_sample, self.n_repl))
        self.mp_ar_stk = np.ones((self.len_sample, self.n_repl))
        self.idx_moist_stk = np.zeros(self.n_repl, dtype=int)
        self.idx_vm_stk = np.zeros(self.n_repl, dtype=int)
        self.moist_ar_stk = np.zeros(self.n_repl)
        self.ash_ar_stk = np.zeros(self.n_repl)
        self.fc_ar_stk = np.zeros(self.n_repl)
        self.vm_ar_stk = np.zeros(self.n_repl)
        self.mp_db_stk = np.ones((self.len_sample, self.n_repl))
        self.ash_db_stk = np.zeros(self.n_repl)
        self.fc_db_stk = np.zeros(self.n_repl)
        self.vm_db_stk = np.zeros(self.n_repl)
        self.mp_daf_stk = np.ones((self.len_sample, self.n_repl))
        self.fc_daf_stk = np.zeros(self.n_repl)
        self.vm_daf_stk = np.zeros(self.n_repl)
        for f, file in enumerate(self.files):
            if TGAExp.T_unit == "Celsius":
                self.T_stk[:, f] = file["T_C"]
            elif TGAExp.T_unit == "Kelvin":
                self.T_stk[:, f] = file["T_K"]
            self.time_stk[:, f] = file["t_min"]

            self.m_ar_stk[:, f] = file["m_mg"]
            self.mp_ar_stk[:, f] = file["m_p"]

            self.idx_moist_stk[f] = np.argmax(self.time_stk[:, f] > self.time_moist + 0.01)

            self.moist_ar_stk[f] = 100 - self.mp_ar_stk[self.idx_moist_stk[f], f]
            self.ash_ar_stk[f] = self.mp_ar_stk[-1, f]
            self.mp_db_stk[:, f] = self.mp_ar_stk[:, f] * 100 / (100 - self.moist_ar_stk[f])
            self.mp_db_stk[:, f] = np.where(self.mp_db_stk[:, f] > 100, 100, self.mp_db_stk[:, f])
            self.ash_db_stk[f] = self.ash_ar_stk[f] * 100 / (100 - self.moist_ar_stk[f])
            self.mp_daf_stk[:, f] = (
                (self.mp_db_stk[:, f] - self.ash_db_stk[f]) * 100 / (100 - self.ash_db_stk[f])
            )
            if self.time_vm is not None:
                self.idx_vm_stk[f] = np.argmax(self.time_stk[:, f] > self.time_vm)
                self.fc_ar_stk[f] = self.mp_ar_stk[self.idx_vm_stk[f], f] - self.ash_ar_stk[f]
                self.vm_ar_stk[f] = (
                    100 - self.moist_ar_stk[f] - self.ash_ar_stk[f] - self.fc_ar_stk[f]
                )
                self.vm_db_stk[f] = self.vm_ar_stk[f] * 100 / (100 - self.moist_ar_stk[f])
                self.fc_db_stk[f] = self.fc_ar_stk[f] * 100 / (100 - self.moist_ar_stk[f])

                self.vm_daf_stk[f] = (
                    (self.vm_db_stk[f] - self.ash_db_stk[f]) * 100 / (100 - self.ash_db_stk[f])
                )
                self.fc_daf_stk[f] = (
                    (self.fc_db_stk[f] - self.ash_db_stk[f]) * 100 / (100 - self.ash_db_stk[f])
                )
        # average
        self.time = np.average(self.time_stk, axis=1)
        self.T = np.average(self.T_stk, axis=1)
        self.T_std = np.std(self.T_stk, axis=1)
        self.mp_ar = np.average(self.mp_ar_stk, axis=1)
        self.mp_ar_std = np.std(self.mp_ar_stk, axis=1)
        self.mp_db = np.average(self.mp_db_stk, axis=1)
        self.mp_db_std = np.std(self.mp_db_stk, axis=1)
        self.moist_ar = np.average(self.moist_ar_stk)
        self.moist_ar_std = np.std(self.moist_ar_stk)
        self.ash_ar = np.average(self.ash_ar_stk)
        self.ash_ar_std = np.std(self.ash_ar_stk)
        self.vm_ar = np.average(self.vm_ar_stk)
        self.vm_ar_std = np.std(self.vm_ar_stk)
        self.fc_ar = np.average(self.fc_ar_stk)
        self.fc_ar_std = np.std(self.fc_ar_stk)

        self.ash_db = np.average(self.ash_db_stk)
        self.ash_db_std = np.std(self.ash_db_stk)
        self.vm_db = np.average(self.vm_db_stk)
        self.vm_db_std = np.std(self.vm_db_stk)
        self.fc_db = np.average(self.fc_db_stk)
        self.fc_db_std = np.std(self.fc_db_stk)
        self.vm_daf = np.average(self.vm_daf_stk)
        self.vm_daf_std = np.std(self.vm_daf_stk)
        self.fc_daf = np.average(self.fc_daf_stk)
        self.fc_daf_std = np.std(self.fc_daf_stk)

        self.len_dtg_db = int((self.Tlims_dtg[1] - self.Tlims_dtg[0]) * TGAExp.resolution_T_dtg)
        self.T_dtg = np.linspace(self.Tlims_dtg[0], self.Tlims_dtg[1], self.len_dtg_db)
        self.time_dtg_stk = np.ones((self.len_dtg_db, self.n_repl))
        self.mp_db_dtg_stk = np.ones((self.len_dtg_db, self.n_repl))
        self.dtg_db_stk = np.ones((self.len_dtg_db, self.n_repl))
        for f, file in enumerate(self.files):
            idxs_dtg = [
                np.argmax(self.T_stk[:, f] > self.Tlims_dtg[0]),
                np.argmax(self.T_stk[:, f] > self.Tlims_dtg[1]),
            ]
            # T_dtg is taken fixed
            self.T_dtg = np.linspace(self.Tlims_dtg[0], self.Tlims_dtg[1], self.len_dtg_db)
            # time start from 0 and consideres a fixed heating rate
            self.time_dtg_stk[:, f] = np.linspace(
                0, self.time_stk[idxs_dtg[1], f] - self.time_stk[idxs_dtg[0], f], self.len_dtg_db
            )

            self.mp_db_dtg_stk[:, f] = np.interp(
                self.T_dtg,
                self.T_stk[idxs_dtg[0] : idxs_dtg[1], f],
                self.mp_db_stk[idxs_dtg[0] : idxs_dtg[1], f],
            )
            # the combusiton indexes use rates as /min
            if TGAExp.dtg_basis == "temperature":
                dtg = np.gradient(self.mp_db_dtg_stk[:, f], self.T_dtg)
            if TGAExp.dtg_basis == "time":
                dtg = np.gradient(self.mp_db_dtg_stk[:, f], self.time_dtg_stk[:, f])
            self.dtg_db_stk[:, f] = SavFil(dtg, TGAExp.dtg_w_SavFil, 1)
        # average
        self.time_dtg = np.average(self.time_dtg_stk, axis=1)
        self.mp_db_dtg = np.average(self.mp_db_dtg_stk, axis=1)
        self.mp_db_dtg_std = np.std(self.mp_db_dtg_stk, axis=1)
        self.dtg_db = np.average(self.dtg_db_stk, axis=1)
        self.dtg_db_std = np.std(self.dtg_db_stk, axis=1)
        self.AveTGstd_p = np.average(self.mp_db_dtg_std)
        self.AveTGstd_p_std = np.nan
        print(
            "Average TG [%] St. Dev. for replicates: "
            + str(round(np.average(self.mp_db_dtg_std), 2))
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
            threshold = np.max(np.abs(self.dtg_db_stk[:, f])) * TGAExp.TiTb_threshold
            # Ti = T at which dtg > Ti_thresh wt%/min after moisture removal
            self.Ti_idx_stk[f] = int(np.argmax(np.abs(self.dtg_db_stk[:, f]) > threshold))
            self.Ti_stk[f] = self.T_dtg[self.Ti_idx_stk[f]]
            # Tp is the T of max abs(dtg)
            self.Tp_idx_stk[f] = int(np.argmax(np.abs(self.dtg_db_stk[:, f])))
            self.Tp_stk[f] = self.T_dtg[self.Tp_idx_stk[f]]
            # Tb reaches < 1 wt%/min at end of curve
            try:
                self.Tb_idx_stk[f] = int(np.flatnonzero(self.dtg_db_stk[:, f] < -threshold)[-1])
            except IndexError:  # the curve nevers goes above 1%
                self.Tb_idx_stk[f] = 0
            self.Tb_stk[f] = self.T_dtg[self.Tb_idx_stk[f]]

            self.dwdT_max_stk[f] = np.max(np.abs(self.dtg_db_stk[:, f]))
            self.dwdT_mean_stk[f] = np.average(np.abs(self.dtg_db_stk[:, f]))
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

    def soliddist_analysis(self, steps_min=[40, 70, 100, 130, 160, 190]):
        """
        Perform solid distance analysis.

        Args:
            steps_min (list, optional): List of minimum steps for analysis.
              Defaults to [40, 70, 100, 130, 160, 190].

        Returns:
            None
        """
        if not self.proximate_computed:
            self.proximate_analysis()
        self.dist_steps_min = steps_min + ["end"]
        len_dist_step = len(self.dist_steps_min)

        self.T_dist_stk = np.ones((len_dist_step, self.n_repl))
        self.time_dist_stk = np.ones((len_dist_step, self.n_repl))
        self.dmp_dist_stk = np.ones((len_dist_step, self.n_repl))
        self.loc_dist_stk = np.ones((len_dist_step, self.n_repl))

        for f, file in enumerate(self.files):
            idxs = []
            for step in steps_min:
                idxs.append(np.argmax(self.time_stk[:, f] > step))
            idxs.append(len(self.time) - 1)
            self.T_dist_stk[:, f] = self.T_stk[idxs, f]
            self.time_dist_stk[:, f] = self.time_stk[idxs, f]

            self.dmp_dist_stk[:, f] = -np.diff(self.mp_db_stk[idxs, f], prepend=100)

            self.loc_dist_stk[:, f] = np.convolve(
                np.insert(self.mp_db_stk[idxs, f], 0, 100), [0.5, 0.5], mode="valid"
            )
        self.T_dist = np.average(self.T_dist_stk, axis=1)
        self.T_dist_std = np.std(self.T_dist_stk, axis=1)
        self.time_dist = np.average(self.time_dist_stk, axis=1)
        self.time_dist_std = np.std(self.time_dist_stk, axis=1)
        self.dmp_dist = np.average(self.dmp_dist_stk, axis=1)
        self.dmp_dist_std = np.std(self.dmp_dist_stk, axis=1)
        self.loc_dist = np.average(self.loc_dist_stk, axis=1)
        self.loc_dist_std = np.std(self.loc_dist_stk, axis=1)
        self.soliddist_computed = True

    def _prepare_deconvolution_model(
        self, centers, sigmas, amplitudes, c_mins, c_maxs, s_mins, s_maxs, a_mins, a_maxs
    ):
        """
        Prepare the deconvolution model for peak fitting.

        Args:
            centers (list): List of peak centers.
            sigmas (list): List of peak sigmas.
            amplitudes (list): List of peak amplitudes.
            c_mins (list): List of minimum values for peak centers.
            c_maxs (list): List of maximum values for peak centers.
            s_mins (list): List of minimum values for peak sigmas.
            s_maxs (list): List of maximum values for peak sigmas.
            a_mins (list): List of minimum values for peak amplitudes.
            a_maxs (list): List of maximum values for peak amplitudes.

        Returns:
            tuple: A tuple containing the deconvolution model and parameters.
        """
        model = LinearModel(prefix="bkg_")
        params = model.make_params(intercept=0, slope=0, vary=False)

        for i, _ in enumerate(centers):
            prefix = f"peak{i}_"
            peak_model = GaussianModel(prefix=prefix)
            pars = peak_model.make_params()
            pars[prefix + "center"].set(value=centers[i], min=c_mins[i], max=c_maxs[i])
            pars[prefix + "sigma"].set(value=sigmas[i], min=s_mins[i], max=s_maxs[i])
            pars[prefix + "amplitude"].set(value=amplitudes[i], min=a_mins[i], max=a_maxs[i])
            model += peak_model
            params.update(pars)

        return model, params

    def deconv_analysis(
        self,
        centers,
        sigmas=None,
        amplitudes=None,
        c_mins=None,
        c_maxs=None,
        s_mins=None,
        s_maxs=None,
        a_mins=None,
        a_maxs=None,
        TLim=None,
    ):
        """
        Perform deconvolution analysis on the data.

        Args:
            centers (list): List of peak centers.
            sigmas (list, optional): List of peak sigmas. Defaults to None.
            amplitudes (list, optional): List of peak amplitudes. Defaults to None.
            c_mins (list, optional): List of minimum values for peak centers. Defaults to None.
            c_maxs (list, optional): List of maximum values for peak centers. Defaults to None.
            s_mins (list, optional): List of minimum values for peak sigmas. Defaults to None.
            s_maxs (list, optional): List of maximum values for peak sigmas. Defaults to None.
            a_mins (list, optional): List of minimum values for peak amplitudes. Defaults to None.
            a_maxs (list, optional): List of maximum values for peak amplitudes. Defaults to None.
            TLim (tuple, optional): Tuple specifying the time range for analysis. Defaults to None.

        Returns:
            None
        """
        if not self.proximate_computed:
            self.proximate_analysis()
        self.dcv_best_fit_stk = np.zeros((self.len_dtg_db, self.n_repl))
        self.dcv_r2_stk = np.zeros(self.n_repl)
        n_peaks = len(centers)
        self.dcv_peaks_stk = np.zeros((self.len_dtg_db, self.n_repl, n_peaks))
        if sigmas is None:
            sigmas = [1] * n_peaks
        if amplitudes is None:
            amplitudes = [10] * n_peaks
        if c_mins is None:
            c_mins = [None] * n_peaks
        if c_maxs is None:
            c_maxs = [None] * n_peaks
        if s_mins is None:
            s_mins = [None] * n_peaks
        if s_maxs is None:
            s_maxs = [None] * n_peaks
        if a_mins is None:
            a_mins = [0] * n_peaks
        if a_maxs is None:
            a_maxs = [None] * n_peaks

        for f in range(self.n_repl):
            y = np.abs(self.dtg_db_stk[:, f])
            model, params = self._prepare_deconvolution_model(
                centers, sigmas, amplitudes, c_mins, c_maxs, s_mins, s_maxs, a_mins, a_maxs
            )
            result = model.fit(y, params=params, x=self.T_dtg)
            self.dcv_best_fit_stk[:, f] = -result.best_fit
            self.dcv_r2_stk[f] = 1 - result.residual.var() / np.var(y)
            components = result.eval_components(x=self.T_dtg)
            for p in range(n_peaks):
                prefix = f"peak{p}_"
                if prefix in components:
                    # Negate the peak data to match the sign of DTG
                    self.dcv_peaks_stk[:, f, p] = -components[prefix]

        self.dcv_best_fit = np.mean(self.dcv_best_fit_stk, axis=1)
        self.dcv_best_fit_std = np.std(self.dcv_best_fit_stk, axis=1)
        self.dcv_r2 = np.mean(self.dcv_r2_stk)
        self.dcv_r2_std = np.std(self.dcv_r2_stk)
        self.dcv_peaks = np.mean(self.dcv_peaks_stk, axis=1)
        self.dcv_peaks_std = np.std(self.dcv_peaks_stk, axis=1)
        self.deconv_computed = True
        # # Plotting the averaged DTG curve and peaks
        self.deconv_plot()

    # section with methods to print reports
    def proximate_report(self):
        """
        Generates a proximate report for the TGA experiment.

        If the proximate analysis has not been computed, it will be computed first.
        The report includes the following columns: 'moist_ar_p', 'ash_ar_p', 'ash_db_p', 'vm_db_p', 'fc_db_p',
        'vm_daf_p', 'fc_daf_p', 'AveTGstd_p'.

        Returns:
            pandas.DataFrame: The proximate report with the calculated values for each sample.
        """

        if not self.proximate_computed:
            self.proximate_analysis()

        out_path = plib.Path(TGAExp.out_path, "SingleSampleReports")
        out_path.mkdir(parents=True, exist_ok=True)

        columns = [
            "moist_ar_p",
            "ash_ar_p",
            "ash_db_p",
            "vm_db_p",
            "fc_db_p",
            "vm_daf_p",
            "fc_daf_p",
            "AveTGstd_p",
        ]
        rep = pd.DataFrame(index=self.filenames, columns=columns)

        for f, filename in enumerate(self.filenames):
            rep.loc[filename] = [
                self.moist_ar_stk[f],
                self.ash_ar_stk[f],
                self.ash_db_stk[f],
                self.vm_db_stk[f],
                self.fc_db_stk[f],
                self.vm_daf_stk[f],
                self.fc_daf_stk[f],
                np.nan,
            ]

        rep.loc["ave"] = [
            self.moist_ar,
            self.ash_ar,
            self.ash_db,
            self.vm_db,
            self.fc_db,
            self.vm_daf,
            self.fc_daf,
            self.AveTGstd_p,
        ]
        rep.loc["std"] = [
            self.moist_ar_std,
            self.ash_ar_std,
            self.ash_db_std,
            self.vm_db_std,
            self.fc_db_std,
            self.vm_daf_std,
            self.fc_daf_std,
            self.AveTGstd_p_std,
        ]
        self.proximate = rep
        rep.to_excel(plib.Path(out_path, self.name + "_proximate.xlsx"))
        self.proximate_report_computed = True
        return self.proximate

    def oxidation_report(self):
        """
        Generates an oxidation report for the TGA experiment.

        If the oxidation analysis has not been computed, it will be computed first.
        The report includes various parameters such as temperature values (Ti, Tp, Tb),
        maximum derivative of weight loss rate (dwdT_max), mean derivative of weight loss rate (dwdT_mean),
        and the combustion sensitivity (S_comb).

        Returns:
            pandas.DataFrame: The oxidation report containing the calculated parameters for each sample.
        """

        if not self.oxidation_computed:
            self.oxidation_analysis()
        out_path = plib.Path(TGAExp.out_path, "SingleSampleReports")
        out_path.mkdir(parents=True, exist_ok=True)
        if TGAExp.T_unit == "Celsius":
            TiTpTb = ["Ti_C", "Tp_C", "Tb_C"]
        elif TGAExp.T_unit == "Kelvin":
            TiTpTb = ["Ti_K", "Tp_K", "Tb_K"]
        columns = TiTpTb + ["idx_dwdT_max_p_min", "dwdT_mean_p_min", "S_comb"]
        rep = pd.DataFrame(index=self.filenames, columns=columns)

        for f, filename in enumerate(self.filenames):
            rep.loc[filename] = [
                self.Ti_stk[f],
                self.Tp_stk[f],
                self.Tb_stk[f],
                self.dwdT_max_stk[f],
                self.dwdT_mean_stk[f],
                self.S_stk[f],
            ]

        rep.loc["ave"] = [self.Ti, self.Tp, self.Tb, self.dwdT_max, self.dwdT_mean, self.S]

        rep.loc["std"] = [
            self.Ti_std,
            self.Tp_std,
            self.Tb_std,
            self.dwdT_max_std,
            self.dwdT_mean_std,
            self.S_std,
        ]
        self.oxidation = rep
        rep.to_excel(plib.Path(out_path, self.name + "_oxidation.xlsx"))
        self.oxidation_report_computed = True
        return self.oxidation

    def soliddist_report(self):
        """
        Generates a report of solid distribution.

        If the solid distribution has not been computed, it computes it using the soliddist_analysis method.
        The report is saved as an Excel file in the 'SingleSampleReports' directory.
        The report includes temperature values and dmp values for each time step.

        Returns:
            pandas.DataFrame: The solid distribution report.
        """
        if not self.soliddist_computed:
            self.soliddist_analysis()
        out_path = plib.Path(TGAExp.out_path, "SingleSampleReports")
        out_path.mkdir(parents=True, exist_ok=True)
        columns = [
            "T [" + TGAExp.T_symbol + "](" + str(s) + "min)" for s in self.dist_steps_min
        ] + ["dmp (" + str(s) + "min)" for s in self.dist_steps_min]
        rep = pd.DataFrame(index=self.filenames, columns=columns)
        for f, filename in enumerate(self.filenames):
            rep.loc[filename] = np.concatenate(
                [self.T_dist_stk[:, f], self.dmp_dist_stk[:, f]]
            ).tolist()
        rep.loc["ave"] = np.concatenate([self.T_dist, self.dmp_dist]).tolist()
        rep.loc["std"] = np.concatenate([self.T_dist_std, self.dmp_dist_std]).tolist()

        self.soliddist = rep
        rep.to_excel(plib.Path(out_path, self.name + "_soliddist.xlsx"))
        self.soliddist_report_computed = True
        return self.soliddist

    # methods to plot results for a single sample
    def tg_plot(self, **kwargs: dict[str, Any]) -> MyFigure:
        """
        Plot the TGA data.

        """
        if not self.proximate_computed:
            self.proximate_analysis()
        out_path = plib.Path(TGAExp.out_path, "single_sample_plots")
        out_path.mkdir(parents=True, exist_ok=True)
        filename = self.name
        mf = MyFigure(
            rows=3,
            cols=1,
            width=4,
            height=10,
            text_font=TGAExp.plot_font,
            x_lab="time [min]",
            y_lab=["T [" + TGAExp.T_symbol + "]", TGAExp.TG_lab + "(stb)", TGAExp.TG_lab + "(db)"],
            grid=TGAExp.plot_grid,
            **kwargs,
        )

        for f in range(self.n_repl):
            mf.axs[0].plot(
                self.time_stk[:, f],
                self.T_stk[:, f],
                color=clrs[f],
                linestyle=lnstls[f],
                label=self.filenames[f],
            )
            mf.axs[1].plot(
                self.time_stk[:, f],
                self.mp_ar_stk[:, f],
                color=clrs[f],
                linestyle=lnstls[f],
                label=self.filenames[f],
            )
            mf.axs[2].plot(
                self.time_stk[:, f],
                self.mp_db_stk[:, f],
                color=clrs[f],
                linestyle=lnstls[f],
                label=self.filenames[f],
            )
            mf.axs[0].vlines(
                self.time_stk[self.idx_moist_stk[f], f],
                self.T_stk[self.idx_moist_stk[f], f] - 50,
                self.T_stk[self.idx_moist_stk[f], f] + 50,
                linestyle=lnstls[f],
                color=clrs[f],
            )
            mf.axs[1].vlines(
                self.time_stk[self.idx_moist_stk[f], f],
                self.mp_ar_stk[self.idx_moist_stk[f], f] - 5,
                self.mp_ar_stk[self.idx_moist_stk[f], f] + 5,
                linestyle=lnstls[f],
                color=clrs[f],
            )
            if self.vm_db < 99:
                mf.axs[0].vlines(
                    self.time_stk[self.idx_vm_stk[f], f],
                    self.T_stk[self.idx_vm_stk[f], f] - 50,
                    self.T_stk[self.idx_vm_stk[f], f] + 50,
                    linestyle=lnstls[f],
                    color=clrs[f],
                )
                mf.axs[2].vlines(
                    self.time_stk[self.idx_vm_stk[f], f],
                    self.mp_db_stk[self.idx_vm_stk[f], f] - 5,
                    self.mp_db_stk[self.idx_vm_stk[f], f] + 5,
                    linestyle=lnstls[f],
                    color=clrs[f],
                )
        mf.save_figure(filename + "_tg", out_path)
        return mf

    def dtg_plot(self):
        """
        Plot the DTG (Derivative Thermogravimetric) data.

        """

        if not self.proximate_computed:
            self.proximate_analysis()
        out_path = plib.Path(TGAExp.out_path, "SingleSamplePlots")
        out_path.mkdir(parents=True, exist_ok=True)

        filename = self.name

        fig, ax, axt, fig_par = figure_create(
            rows=3, cols=1, plot_type=0, paper_col=1, font=TGAExp.plot_font
        )
        for f in range(self.n_repl):
            ax[0].plot(
                self.time_dtg,
                self.T_dtg,
                color=clrs[f],
                linestyle=lnstls[f],
                label=self.filenames[f],
            )
            ax[1].plot(self.time_dtg, self.mp_db_dtg_stk[:, f], color=clrs[f], linestyle=lnstls[f])
            ax[2].plot(self.time_dtg, self.dtg_db_stk[:, f], color=clrs[f], linestyle=lnstls[f])
            if self.oxidation_computed:
                ax[2].vlines(
                    self.time_dtg[self.Ti_idx_stk[f]],
                    ymin=-1.5,
                    ymax=0,
                    linestyle=lnstls[f],
                    color=clrs[f],
                    label="Ti",
                )
                ax[2].vlines(
                    self.time_dtg[self.Tp_idx_stk[f]],
                    ymin=np.min(self.dtg_db_stk[:, f]),
                    ymax=np.min(self.dtg_db_stk[:, f]) / 5,
                    linestyle=lnstls[f],
                    color=clrs[f],
                    label="Tp",
                )
                ax[2].vlines(
                    self.time_dtg[self.Tb_idx_stk[f]],
                    ymin=-1.5,
                    ymax=0,
                    linestyle=lnstls[f],
                    color=clrs[f],
                    label="Tb",
                )
        ax[0].legend(loc="best")
        figure_save(
            filename + "_dtg",
            out_path,
            fig,
            ax,
            axt,
            fig_par,
            x_lab="time [min]",
            y_lab=["T [" + TGAExp.T_symbol + "]", TGAExp.TG_lab + "(db)", TGAExp.DTG_lab + "(db)"],
            grid=TGAExp.plot_grid,
        )

    def soliddist_plot(self, paper_col=1, hgt_mltp=1.25):
        """
        Plot the solid distribution analysis.

        Args:
            paper_col (int): Number of columns in the plot for paper publication (default is 1).
            hgt_mltp (float): Height multiplier for the plot (default is 1.25).

        Returns:
            None
        """

        # slightly different plotting behaviour (uses averages)
        if not self.soliddist_computed:
            self.soliddist_analysis()
        out_path = plib.Path(TGAExp.out_path, "SingleSamplePlots")
        out_path.mkdir(parents=True, exist_ok=True)
        filename = self.name
        fig, ax, axt, fig_par = figure_create(
            rows=2,
            cols=1,
            plot_type=0,
            paper_col=paper_col,
            hgt_mltp=hgt_mltp,
            font=TGAExp.plot_font,
        )

        ax[0].plot(self.time, self.T)
        ax[0].fill_between(self.time, self.T - self.T_std, self.T + self.T_std, alpha=0.3)
        ax[1].plot(self.time, self.mp_db)
        ax[1].fill_between(
            self.time, self.mp_db - self.mp_db_std, self.mp_db + self.mp_db_std, alpha=0.3
        )
        for tm, mp, dmp in zip(self.time_dist, self.loc_dist, self.dmp_dist):
            ax[1].annotate(
                str(np.round(dmp, 0)) + "%", ha="center", va="top", xy=(tm - 10, mp + 1), fontsize=9
            )
        figure_save(
            filename + "_soliddist",
            out_path,
            fig,
            ax,
            axt,
            fig_par,
            x_lab="time [min]",
            y_lab=["T [" + TGAExp.T_symbol + "]", TGAExp.TG_lab + "(db)"],
            grid=TGAExp.plot_grid,
        )

    def deconv_plot(
        self,
        filename="Deconv",
        x_lim=None,
        y_lim=None,
        save_as_pdf=False,
        save_as_svg=False,
        legend="best",
    ):
        """
        Plot the deconvolution results.

        Args:
            filename (str, optional): The filename to save the plot. Defaults to 'Deconv'.
            x_lim (tuple, optional): The x-axis limits of the plot. Defaults to None.
            y_lim (tuple, optional): The y-axis limits of the plot. Defaults to None.
            save_as_pdf (bool, optional): Whether to save the plot as a PDF file. Defaults to False.
            save_as_svg (bool, optional): Whether to save the plot as an SVG file. Defaults to False.
            legend (str, optional): The position of the legend in the plot. Defaults to 'best'.
        """

        if not self.deconv_computed:
            raise Exception("Deconvolution analysis not computed")
        out_path_dcv = plib.Path(TGAExp.out_path, "SingleSampleDeconvs")
        out_path_dcv.mkdir(parents=True, exist_ok=True)
        filename = self.name
        fig, ax, axt, fig_par = figure_create(
            rows=1, cols=1, plot_type=0, paper_col=0.78, hgt_mltp=1.25, font=TGAExp.plot_font
        )
        # Plot DTG data
        ax[0].plot(self.T_dtg, self.dtg_db, color="black", label="DTG")
        ax[0].fill_between(
            self.T_dtg,
            self.dtg_db - self.dtg_db_std,
            self.dtg_db + self.dtg_db_std,
            color="black",
            alpha=0.3,
        )

        # Plot best fit and individual peaks
        ax[0].plot(self.T_dtg, self.dcv_best_fit, label="best fit", color="red", linestyle="--")
        ax[0].fill_between(
            self.T_dtg,
            self.dcv_best_fit - self.dcv_best_fit_std,
            self.dcv_best_fit + self.dcv_best_fit_std,
            color="red",
            alpha=0.3,
        )
        clrs_p = clrs[:3] + clrs[5:]  # avoid using red
        p = 0
        for peak, peak_std in zip(self.dcv_peaks.T, self.dcv_peaks_std.T):

            ax[0].plot(
                self.T_dtg,
                peak,
                label="peak " + str(int(p + 1)),
                color=clrs_p[p],
                linestyle=lnstls[p],
            )
            ax[0].fill_between(
                self.T_dtg, peak - peak_std, peak + peak_std, color=clrs_p[p], alpha=0.3
            )
            p += 1
        ax[0].annotate(
            f"r$^2$={self.dcv_r2:.2f}", xycoords="axes fraction", xy=(0.85, 0.96), size="x-small"
        )

        # Save figure using figure_save
        figure_save(
            filename,
            out_path_dcv,
            fig,
            ax,
            axt,
            fig_par,
            x_lab="T [" + TGAExp.T_symbol + "]",
            y_lab=TGAExp.DTG_lab,
            x_lim=x_lim,
            y_lim=y_lim,
            legend=legend,
            grid=TGAExp.plot_grid,
            save_as_pdf=save_as_pdf,
            save_as_svg=save_as_svg,
        )  # Set additional parameters as needed
