# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:28:04 2022

@author: mp933
"""
# %%
from __future__ import annotations
import pathlib as plib
import numpy as np
import pandas as pd
from typing import Any
from scipy.signal import savgol_filter
from lmfit.models import GaussianModel, LinearModel
from typing import Literal
from tga_data_analysis.myfigure import MyFigure, clrs, lnstls, htchs, mrkrs


class Project:

    def __init__(
        self,
        folder_path: plib.Path,
        name: str | None = None,
        temp_unit: Literal["C", "K"] = "C",
        plot_font: Literal["Dejavu Sans", "Times New Roman"] = "Dejavu Sans",
        dtg_basis: Literal["temperature", "time"] = "temperature",
        temp_i_temp_b_threshold: float = 0.01,  # % of the peak that is used for Ti and Tb
        resolution_sec_deg_dtg: int = 5,
        dtg_window_filter: int = 101,
        plot_grid: bool = False,
        column_name_mapping: dict[str, str] | None = None,
        load_skiprows: int = 0,
        time_moist: float = 38.0,
        time_vm: float = 147.0,
        temp_initial_celsius: float = 40,
        auto_save_reports: bool = True,
    ):
        self.folder_path = folder_path
        self.out_path = plib.Path(folder_path, "output")
        if name is None:
            self.name = self.folder_path.parts[-1]
        else:
            self.name = name
        self.temp_unit = temp_unit
        self.plot_font = plot_font
        self.plot_grid = plot_grid
        self.dtg_basis = dtg_basis
        self.temp_i_temp_b_threshold = temp_i_temp_b_threshold
        self.resolution_sec_deg_dtg = resolution_sec_deg_dtg
        self.dtg_window_filter = dtg_window_filter
        self.load_skiprows = load_skiprows
        self.time_moist = time_moist
        self.time_vm = time_vm
        self.temp_initial_celsius = temp_initial_celsius
        self.auto_save_reports = auto_save_reports

        if self.temp_unit == "C":
            self.temp_symbol = "°C"
        elif self.temp_unit == "K":
            self.temp_symbol = "K"

        self.tg_label = "TG [wt%]"
        if self.dtg_basis == "temperature":
            self.dtg_label = "DTG [wt%/" + self.temp_symbol + "]"
        elif self.dtg_basis == "time":
            self.dtg_label = "DTG [wt%/min]"

        if column_name_mapping is None:
            self.column_name_mapping = {
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
        else:
            self.column_name_mapping = column_name_mapping
        #
        self.samples: dict[str, Sample] = {}
        self.samplenames: list[str] = []

        self.multireports: dict[str, pd.DataFrame] = {}
        self.multireport_types_computed: list[str] = []

    def add_sample(self, samplename: str, sample: Sample):
        if samplename not in self.samplenames:
            self.samplenames.append(samplename)
            self.samples[samplename] = sample
        else:
            raise ValueError(f"{samplename = } already used")

    def multireport(
        self,
        samplenames: list[str] | None = None,
        labels: list[str] | None = None,
        report_type: Literal[
            "proximate", "oxidation", "oxidation_extended", "soliddist", "soliddist_extended"
        ] = "proximate",
        report_style: Literal["repl_ave_std", "ave_std", "ave_pm_std"] = "ave_std",
        decimals_in_ave_pm_std: int = 2,
        filename: str | None = None,
    ) -> pd.DataFrame:
        """
        Generate a multi-sample proximate report.

        Args:
            exps (list): List of experiments.
            filename (str, optional): Name of the output file. Defaults to 'Rep'.

        Returns:
            pandas.DataFrame: DataFrame containing the multi-sample proximate report.
        """
        if samplenames is None:
            samplenames = self.samplenames

        samples = [self.samples[samplename] for samplename in samplenames]

        if labels is None:
            labels = samplenames
        for sample in samples:
            if report_type not in sample.report_types_computed:
                sample.report(report_type)

        if report_type == "soliddist":
            reports = [sample.reports[report_type] for sample in samples]
            reports = self._reformat_ave_std_columns(reports)
        elif report_type == "soliddist_extended":
            raise ValueError(
                f"{report_type = } not allowed for multireport, use 'soliddist' instead"
            )
        else:
            reports = [sample.reports[report_type] for sample in samples]

        if report_style == "repl_ave_std":
            # Concatenate all individual reports
            report = pd.concat(reports, keys=labels)
            report.index.names = [None, None]  # Remove index names

        elif report_style == "ave_std":
            # Keep only the average and standard deviation
            ave_std_dfs = []
            for label, report in zip(labels, reports):
                ave_std_dfs.append(report.loc[["ave", "std"]])
            report = pd.concat(ave_std_dfs, keys=labels)
            report.index.names = [None, None]  # Remove index names

        elif report_style == "ave_pm_std":
            # Format as "ave ± std" and use sample name as the index
            rows = []
            for label, report in zip(labels, reports):
                row = {
                    col: f"{report.at['ave', col]:.{decimals_in_ave_pm_std}f} ± {report.at['std', col]:.{decimals_in_ave_pm_std}f}"
                    for col in report.columns
                }
                rows.append(pd.Series(row, name=label))
            report = pd.DataFrame(rows)

        else:
            raise ValueError(f"{report_style = } is not a valid option")
        self.multireport_types_computed.append(report_type)
        self.multireports[report_type] = report
        if self.auto_save_reports:
            out_path = plib.Path(self.out_path, "multireports")
            out_path.mkdir(parents=True, exist_ok=True)
            if filename is None:
                filename = f"{self.name}_{report_type}_{report_style}.xlsx"
            else:
                filename = filename + ".xlsx"
            report.to_excel(plib.Path(out_path, filename))
        return report

    def plot_multireport(
        self,
        filename: str = "plot",
        samplenames: list[str] | None = None,
        labels: list[str] | None = None,
        report_type: Literal["proximate", "oxidation", "soliddist"] = "proximate",
        bar_labels: list[str] | None = None,
        **kwargs,
    ) -> MyFigure:
        """
        Generate a multi-plot for proximate analysis.

        Parameters:
        - exps (list): List of experiments.
        - filename (str): Name of the output file (default: "Prox").
        - smpl_labs (list): List of sample labels (default: None).
        - xlab_rot (int): Rotation angle of x-axis labels (default: 0).
        - paper_col (float): Color of the plot background (default: 0.8).
        - hgt_mltp (float): Height multiplier of the plot (default: 1.5).
        - bboxtoanchor (bool): Whether to place the legend outside the plot area (default: True).
        - x_anchor (float): X-coordinate of the legend anchor point (default: 1.13).
        - y_anchor (float): Y-coordinate of the legend anchor point (default: 1.02).
        - legend_loc (str): Location of the legend (default: 'best').
        - y_lim (list): Y-axis limits for the bar plot (default: [0, 100]).
        - yt_lim (list): Y-axis limits for the scatter plot (default: [0, 1]).
        - y_ticks (list): Y-axis tick positions for the bar plot (default: None).
        - yt_ticks (list): Y-axis tick positions for the scatter plot (default: None).

        Returns:
        - None
        """
        kw_keys = ["height", "width", "grid", "text_font"]
        kw_default_values = [
            4,
            4,
            self.plot_grid,
            self.plot_font,
        ]
        for kwk, kwd in zip(kw_keys, kw_default_values):
            if kwk not in kwargs.keys():
                kwargs[kwk] = kwd

        df = self.multireport(samplenames, labels, report_type, report_style="ave_std")
        df_ave = df.xs("ave", level=1, drop_level=False)
        df_std = df.xs("std", level=1, drop_level=False)
        # drop the multi-level index to simplify the DataFrame
        df_ave = df_ave.droplevel(1)
        df_std = df_std.droplevel(1)
        out_path = plib.Path(self.out_path, "multireport_plots")
        out_path.mkdir(parents=True, exist_ok=True)

        if report_type == "proximate":
            vars_bar = ["moisture (stb)", "VM (db)", "FC (db)", "ash (db)"]
            df_ave.columns = vars_bar
            df_std.columns = vars_bar
            bar_yaxis = vars_bar
            bar_ytaxis = None
            twinx = None
            y_lab = self.tg_label
            yt_lab = None

        elif report_type == "oxidation":
            vars_bar = ["T$_i$", "T$_p$", "T$_b$", "S"]
            df_ave.columns = vars_bar
            df_std.columns = vars_bar
            bar_yaxis = ["T$_i$", "T$_p$", "T$_b$"]
            bar_ytaxis = "S"
            twinx = True
            y_lab = f"T [{self.temp_symbol}]"
            yt_lab = "S (comb. index)"

        elif report_type == "soliddist":
            vars_bar = [f"{col.split(" ")[0]} {col.split(" ")[-1]}" for col in df_ave.columns]
            df_ave.columns = vars_bar
            df_std.columns = vars_bar
            bar_yaxis = vars_bar
            bar_ytaxis = None
            twinx = None
            y_lab = "step mass loss [wt%]"
            yt_lab = None

        myfig = MyFigure(
            rows=1,
            cols=1,
            twinx=twinx,
            y_lab=y_lab,
            yt_lab=yt_lab,
            **kwargs,
        )
        df_ave[bar_yaxis].plot(
            ax=myfig.axs[0],
            kind="bar",
            yerr=df_std[bar_yaxis],
            capsize=2,
            width=0.85,
            ecolor="k",
            edgecolor="black",
        )
        if bar_ytaxis is not None:
            myfig.axts[0].scatter(
                df_ave.index,
                df_ave[bar_ytaxis],
                label=bar_ytaxis,
                edgecolor="black",
                color=clrs[3],
                # s=100
            )
            myfig.axts[0].errorbar(
                df_ave.index,
                df_ave[bar_ytaxis],
                yerr=df_std[bar_ytaxis],
                ecolor="k",
                linestyle="None",
                capsize=2,
            )
        myfig.save_figure(filename + report_type, out_path)
        return myfig

    def plot_multi_tg(
        self,
        filename: str = "plot",
        samplenames: list[str] | None = None,
        labels: list[str] | None = None,
        **kwargs,
    ) -> MyFigure:
        """
        Plot multiple thermogravimetric (TG) curves.

        Args:
            exps (list): List of experimental data objects.
            filename (str, optional): Name of the output file. Defaults to 'Fig'.
            paper_col (float, optional): Width of the figure in inches. Defaults to 0.78.
            hgt_mltp (float, optional): Height multiplier of the figure. Defaults to 1.25.
            x_lim (tuple, optional): Limits of the x-axis. Defaults to None.
            y_lim (list, optional): Limits of the y-axis. Defaults to [0, 100].
            y_ticks (list, optional): Custom y-axis tick locations. Defaults to None.
            lttrs (bool, optional): Whether to annotate letters on the plot. Defaults to False.
            save_as_pdf (bool, optional): Whether to save the figure as a PDF file. Defaults to False.
            save_as_svg (bool, optional): Whether to save the figure as an SVG file. Defaults to False.

        Returns:
            None
        """
        if samplenames is None:
            samplenames = self.samplenames

        samples = [self.samples[samplename] for samplename in samplenames]

        if labels is None:
            labels = samplenames
        for sample in samples:
            if not sample.proximate_computed:
                sample.proximate_analysis()

        keys = ["height", "width", "grid", "text_font", "x_lab", "y_lab"]
        values = [
            4,
            4,
            self.plot_grid,
            self.plot_font,
            f"T [{self.temp_symbol}]",
            self.tg_label,
        ]
        for kwk, kwd in zip(keys, values):
            if kwk not in kwargs.keys():
                kwargs[kwk] = kwd

        out_path = plib.Path(self.out_path, "multisample_plots")
        out_path.mkdir(parents=True, exist_ok=True)
        myfig = MyFigure(
            rows=1,
            cols=1,
            **kwargs,
        )
        for i, sample in enumerate(samples):
            myfig.axs[0].plot(
                sample.temp.ave(),
                sample.mp_db.ave(),
                color=clrs[i],
                linestyle=lnstls[i],
                label=labels[i],
            )
            myfig.axs[0].fill_between(
                sample.temp.ave(),
                sample.mp_db.ave() - sample.mp_db.std(),
                sample.mp_db.ave() + sample.mp_db.std(),
                color=clrs[i],
                alpha=0.3,
            )
        myfig.save_figure(filename, out_path)
        return myfig

    def kas_analysis(
        self,
        samplenames: list[str] | None = None,
        ramps: list[float] | None = None,
        alpha: list[float] | None = None,
    ):
        """
        Perform KAS (Kissinger-Akahira-Sunose) analysis on a set of experiments.

        Args:
            samplenames (list[str]): List of sample names to analyze.
            ramps (list[float]): List of ramp values used for each experiment.
            alpha (list[float]): List of alpha values to investigate. Defaults to np.arange(0.05, .9, 0.05).

        Returns:
            dict: Results of the KAS analysis.
        """
        if samplenames is None:
            samplenames = self.samplenames

        samples = [self.samples[name] for name in samplenames if name in self.samples]
        if ramps is None:
            ramps = [sample.heating_rate_deg_min for sample in samples]
        if alpha is None:
            alpha = np.arange(0.05, 0.9, 0.05)

        r_gas_constant = 8.314462618  # Universal gas constant in J/(mol*K)
        c_to_k = 273.15
        activation_energies = np.zeros(len(alpha))
        std_devs = np.zeros(len(alpha))
        fits = []
        x_matrix = np.zeros((len(alpha), len(ramps)))
        y_matrix = np.zeros((len(alpha), len(ramps)))

        for idx, sample in enumerate(samples):
            temp = sample.temp_dtg + c_to_k if sample.temp_unit == "C" else sample.temp_dtg
            converted_mass = (sample.mp_db_dtg() - np.min(sample.mp_db_dtg())) / (
                np.max(sample.mp_db_dtg()) - np.min(sample.mp_db_dtg())
            )
            alpha_mass = 1 - converted_mass

            for alpha_idx, alpha_val in enumerate(alpha):
                conversion_index = np.argmax(alpha_mass > alpha_val)
                x_matrix[alpha_idx, idx] = 1 / temp[conversion_index] * 1000
                y_matrix[alpha_idx, idx] = np.log(ramps[idx] / temp[conversion_index] ** 2)

        for i in range(len(alpha)):
            p, cov = np.polyfit(x_matrix[i, :], y_matrix[i, :], 1, cov=True)
            fits.append(np.poly1d(p))
            activation_energies[i] = -p[1] * r_gas_constant
            std_devs[i] = np.sqrt(cov[0][0]) * r_gas_constant

        shared_kas_analysis_name = "".join(
            [
                char
                for char in samplenames[0]
                if all(char in samplename for samplename in samplenames)
            ]
        )
        kas_result = {
            "Ea": activation_energies,
            "Ea_std": std_devs,
            "alpha": alpha,
            "ramps": ramps,
            "x_matrix": x_matrix,
            "y_matrix": y_matrix,
            "fits": fits,
            "name": shared_kas_analysis_name,
        }

        for sample in samples:
            sample.kas = kas_result

        return kas_result

    def _reformat_ave_std_columns(self, reports):
        """
        Reformat column names based on the average and standard deviation.

        Args:
            reports (list of pd.DataFrame): List of reports to reformat column names.
        """
        # Check that all reports have the same number of columns
        num_columns = len(reports[0].columns)
        if not all(len(report.columns) == num_columns for report in reports):
            raise ValueError("All reports must have the same number of columns.")

        # Initialize a list to hold the new formatted column names
        formatted_column_names = []

        # Iterate over each column index
        for i in range(num_columns):
            # Extract the numeric part of the column name (assume it ends with ' K' or ' C')
            column_values = [float(report.columns[i].split()[0]) for report in reports]
            ave = np.mean(column_values)
            std = np.std(column_values)

            # Determine the unit (assuming all columns have the same unit)
            unit = reports[0].columns[i].split()[-1]

            # Create the new column name with the unit
            formatted_column_name = f"{ave:.0f} ± {std:.0f} {unit}"
            formatted_column_names.append(formatted_column_name)

        # Rename the columns in each report using the new formatted names
        for report in reports:
            report.columns = formatted_column_names

        return reports


class Sample:

    def __init__(
        self,
        project: Project,
        name: str,
        filenames: list[str],
        folder_path: plib.Path | None = None,
        label: str | None = None,
        correct_ash_mg: list[float] | None = None,
        correct_ash_fr: list[float] | None = None,
        column_name_mapping: dict[str:str] | None = None,
        load_skiprows: int = 0,
        time_moist: float = 38.0,
        time_vm: float = 147,
        temp_lim_dtg_celsius: tuple[float] | None = None,
        heating_rate_deg_min: float | None = None,
    ):
        # store the sample in the project
        self.project_name = project.name
        project.add_sample(name, self)
        # prject defaults unless specified

        self.out_path = project.out_path
        self.temp_unit = project.temp_unit
        self.temp_symbol = project.temp_symbol
        self.tg_label = project.tg_label
        self.dtg_label = project.dtg_label
        self.plot_font = project.plot_font
        self.plot_grid = project.plot_grid
        self.dtg_basis = project.dtg_basis
        self.temp_lim_dtg = project.temp_lim_dtg
        self.len_dtg_db = project.len_dtg_db
        self.temp_dtg = project.temp_dtg
        self.auto_save_reports = project.auto_save_reports
        self.temp_i_temp_b_threshold = project.temp_i_temp_b_threshold
        self.resolution_sec_deg_dtg = project.resolution_sec_deg_dtg
        self.dtg_window_filter = project.dtg_window_filter
        self.temp_initial_celsius = project.temp_initial_celsius
        self.dtg_window_filter = project.dtg_window_filter
        if folder_path is None:
            self.folder_path = project.folder_path
        else:
            self.folder_path = folder_path
        if column_name_mapping is None:
            self.column_name_mapping = project.column_name_mapping
        else:
            self.column_name_mapping = column_name_mapping
        if load_skiprows is None:
            self.load_skiprows = project.load_skiprows
        else:
            self.load_skiprows = load_skiprows
        if time_moist is None:
            self.time_moist = project.time_moist
        else:
            self.time_moist = time_moist
        if time_vm is None:
            self.time_vm = project.time_vm
        else:
            self.time_vm = time_vm
        # sample default
        self.name = name
        self.filenames = filenames
        self.n_repl = len(self.filenames)
        self.heating_rate_deg_min = heating_rate_deg_min
        self.correct_ash_mg = self._broadcast_value_prop(correct_ash_mg)
        self.correct_ash_fr = self._broadcast_value_prop(correct_ash_fr)
        if not label:
            self.label = name
        else:
            self.label = label

        if temp_lim_dtg_celsius is None:
            self.temp_lim_dtg_celsius = [120, 880]
        else:
            self.temp_lim_dtg_celsius = temp_lim_dtg_celsius
        if self.temp_unit == "C":
            self.temp_lim_dtg = self.temp_lim_dtg_celsius
        elif self.temp_unit == "K":
            self.temp_lim_dtg = [t + 273.15 for t in self.temp_lim_dtg_celsius]
        else:
            raise ValueError(f"{self.temp_unit = } is not acceptable")

        self.len_dtg_db: int = int(
            (self.temp_lim_dtg[1] - self.temp_lim_dtg[0]) * project.resolution_sec_deg_dtg
        )
        self.temp_dtg: np.ndarray = np.linspace(
            self.temp_lim_dtg[0], self.temp_lim_dtg[1], self.len_dtg_db
        )
        # for variables and computations
        self.files: dict[str : pd.DataFrame] = {}
        self.len_files: dict[str : pd.DataFrame] = {}
        self.len_sample: int = 0

        # proximate
        self.temp: Measure = Measure(name="temp_" + self.temp_unit)
        self.time: Measure = Measure(name="time")
        self.m_ar: Measure = Measure(name="m_ar")
        self.mp_ar: Measure = Measure(name="mp_ar")
        self.idx_moist: Measure = Measure(name="idx_moist")
        self.idx_vm: Measure = Measure(name="idx_vm")
        self.moist_ar: Measure = Measure(name="moist_ar")
        self.ash_ar: Measure = Measure(name="ash_ar")
        self.fc_ar: Measure = Measure(name="fc_ar")
        self.vm_ar: Measure = Measure(name="vm_ar")
        self.mp_db: Measure = Measure(name="mp_db")
        self.ash_db: Measure = Measure(name="ash_db")
        self.fc_db: Measure = Measure(name="fc_db")
        self.vm_db: Measure = Measure(name="vm_db")
        self.mp_daf: Measure = Measure(name="mp_daf")
        self.fc_daf: Measure = Measure(name="fc_daf")
        self.vm_daf: Measure = Measure(name="vm_daf")
        self.time_dtg: Measure = Measure(name="time_dtg")
        self.mp_db_dtg: Measure = Measure(name="mp_db_dtg")
        self.dtg_db: Measure = Measure(name="dtg_db")
        self.ave_dev_tga_perc: float | None = None
        # oxidation
        self.temp_i_idx: Measure = Measure(name="temp_i_idx")
        self.temp_i: Measure = Measure(name="temp_i_" + self.temp_unit)
        self.temp_p_idx: Measure = Measure(name="temp_p_idx")
        self.temp_p: Measure = Measure(name="temp_p_" + self.temp_unit)
        self.temp_b_idx: Measure = Measure(name="temp_b_idx")
        self.temp_b: Measure = Measure(name="temp_b_" + self.temp_unit)
        self.dwdtemp_max: Measure = Measure(name="dwdtemp_max")
        self.dwdtemp_mean: Measure = Measure(name="dwdtemp_mean")
        self.s_combustion_index: Measure = Measure(name="s_combustion_index")
        # soliddist
        self.temp_soliddist: Measure = Measure(name="temp_dist_" + self.temp_unit)
        self.time_soliddist: Measure = Measure(name="time_dist")
        self.dmp_soliddist: Measure = Measure(name="dmp_dist")
        self.loc_soliddist: Measure = Measure(name="loc_dist")
        # deconvolution
        self.dcv_best_fit: Measure = Measure(name="dcv_best_fit")
        self.dcv_r2: Measure = Measure(name="dcv_r2")
        self.dcv_peaks: list[Measure] = []
        # Flag to track if data is loaded
        self.proximate_computed = False
        self.files_loaded = False
        self.oxidation_computed = False
        self.soliddist_computed = False
        self.deconv_computed = False
        self.KAS_computed = False
        # for reports
        self.reports: dict[str, pd.DataFrame] = {}
        self.report_types_computed: list[str] = []

        self.load_files()
        self.proximate_analysis()

    def _broadcast_value_prop(self, prop: list | str | float | int | bool) -> list:
        """_summary_

        :param prop: _description_
        :type prop: list | str | float | int | bool
        :raises ValueError: _description_
        :return: _description_
        :rtype: list
        """
        if prop is None:
            broad_prop = [None] * self.n_repl
        if isinstance(prop, (list, tuple)):
            # If it's a list or tuple, but we're not expecting pairs, it's a single value per axis.
            if len(prop) == self.n_repl:
                broad_prop = prop
            else:
                raise ValueError(
                    f"The size of the property '{prop}' does not match the number of replicates."
                )
        if isinstance(prop, (str, float, int, bool)):
            broad_prop = [prop] * self.n_repl
        return broad_prop

    def load_single_file(
        self,
        filename: str,
        folder_path: plib.Path | None = None,
        load_skiprows: int | None = None,
        column_name_mapping: dict | None = None,
    ) -> pd.DataFrame:
        if column_name_mapping is None:
            column_name_mapping = self.column_name_mapping
        if folder_path is None:
            folder_path = self.folder_path
        if load_skiprows is None:
            load_skiprows = self.load_skiprows
        file_path = plib.Path(folder_path, filename + ".txt")
        if not file_path.is_file():
            file_path = plib.Path(folder_path, filename + ".csv")
        file = pd.read_csv(file_path, sep="\t", skiprows=load_skiprows)
        if file.shape[1] < 3:
            file = pd.read_csv(file_path, sep=",", skiprows=load_skiprows)
        file = file.rename(columns={col: column_name_mapping.get(col, col) for col in file.columns})
        for column in file.columns:
            file[column] = pd.to_numeric(file[column], errors="coerce")
        file.dropna(inplace=True)
        return file

    def correct_file_values(
        self,
        file: pd.DataFrame,
        correct_ash_fr: float | None,
        correct_ash_mg: float | None,
    ):
        if correct_ash_mg is not None:
            file["m_mg"] = file["m_mg"] - np.min(file["m_mg"]) + correct_ash_mg
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
        if correct_ash_fr is not None:
            file["m_p"] = file["m_p"] - np.min(file["m_p"]) + correct_ash_fr
            file["m_p"] = file["m_p"] / np.max(file["m_p"]) * 100
        file = file[file["T_C"] >= self.temp_initial_celsius].copy()
        file["T_K"] = file["T_C"] + 273.15
        return file

    def load_files(self):
        """
        Loads all the files for the experiment.

        Returns:
            list: The list of loaded files.
        """
        print("\n" + self.name)
        # import files and makes sure that replicates have the same size
        for f, filename in enumerate(self.filenames):
            print(filename)
            file_to_correct = self.load_single_file(filename)

            file = self.correct_file_values(
                file_to_correct, self.correct_ash_fr[f], self.correct_ash_fr[f]
            )
            # FILE CORRECTION
            self.files[filename] = file
            self.len_files[filename] = max(file.shape)
        self.len_sample = min(self.len_files.values())
        # keep the shortest vector size for all replicates, create the object
        for filename in self.filenames:
            self.files[filename] = self.files[filename].head(self.len_sample)
        self.files_loaded = True  # Flag to track if data is loaded
        return self.files

    def proximate_analysis(self):
        """
        Performs proximate analysis on the loaded data.
        """
        if not self.files_loaded:
            self.load_files()

        for f, file in enumerate(self.files.values()):
            if self.temp_unit == "C":
                self.temp.add(f, file["T_C"])
            elif self.temp_unit == "K":
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
            self.dtg_db.add(f, savgol_filter(dtg, self.dtg_window_filter, 1))
        # average
        self.ave_dev_tga_perc = np.average(self.mp_db_dtg.std())
        print(f"Average TG [%] St. Dev. for replicates: {self.ave_dev_tga_perc:0.2f} %")
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

        for f in range(self.n_repl):
            threshold: float = np.max(np.abs(self.dtg_db.stk(f))) * self.temp_i_temp_b_threshold
            # Ti = T at which dtg > Ti_thresh wt%/min after moisture removal
            self.temp_i_idx.add(f, int(np.argmax(np.abs(self.dtg_db.stk(f)) > threshold)))
            self.temp_i.add(f, self.temp_dtg[self.temp_i_idx.stk(f)])
            # Tp is the T of max abs(dtg)
            self.temp_p_idx.add(f, int(np.argmax(np.abs(self.dtg_db.stk(f)))))
            self.temp_p.add(f, self.temp_dtg[self.temp_p_idx.stk(f)])
            # Tb reaches < 1 wt%/min at end of curve
            try:
                self.temp_b_idx.add(f, int(np.flatnonzero(self.dtg_db.stk(f) < -threshold)[-1]))
            except IndexError:  # the curve nevers goes above 1%
                self.temp_b_idx.add(f, 0)
            self.temp_b.add(f, self.temp_dtg[self.temp_b_idx.stk(f)])

            self.dwdtemp_max.add(f, np.max(np.abs(self.dtg_db.stk(f))))
            self.dwdtemp_mean.add(f, np.average(np.abs(self.dtg_db.stk(f))))
            # combustion index
            self.s_combustion_index.add(
                f,
                (
                    self.dwdtemp_max.stk(f)
                    * self.dwdtemp_mean.stk(f)
                    / self.temp_i.stk(f)
                    / self.temp_i.stk(f)
                    / self.temp_b.stk(f)
                ),
            )
        # # average
        self.oxidation_computed = True

    def soliddist_analysis(self, steps_min: list[float] | None = None):
        """
        Perform solid distance analysis.

        Args:
            steps_min (list, optional): List of minimum steps for analysis.
              Defaults to [40, 70, 100, 130, 160, 190].

        Returns:
            None
        """
        if steps_min is None:
            steps_min = [40, 70, 100, 130, 160, 190]
        if not self.proximate_computed:
            self.proximate_analysis()

        for f in range(self.n_repl):
            idxs = []
            for step in steps_min:
                idxs.append(np.argmax(self.time.stk(f) > step))
            idxs.append(len(self.time.stk(f)) - 1)
            self.temp_soliddist.add(f, self.temp.stk(f)[idxs])
            self.time_soliddist.add(f, self.time.stk(f)[idxs])

            self.dmp_soliddist.add(f, -np.diff(self.mp_db.stk(f)[idxs], prepend=100))

            self.loc_soliddist.add(
                f, np.convolve(np.insert(self.mp_db.stk(f)[idxs], 0, 100), [0.5, 0.5], mode="valid")
            )

        self.soliddist_computed = True

    def deconv_analysis(
        self,
        centers: list[float],
        sigmas: list[float] | None = None,
        amplitudes: list[float] | None = None,
        center_mins: list[float] | None = None,
        center_maxs: list[float] | None = None,
        sigma_mins: list[float] | None = None,
        sigma_maxs: list[float] | None = None,
        amplitude_mins: list[float] | None = None,
        amplitude_maxs: list[float] | None = None,
    ):
        """
        Perform deconvolution analysis on the data.

        Args:
            centers (list): List of peak centers.
            sigmas (list, optional): List of peak sigmas. Defaults to None.
            amplitudes (list, optional): List of peak amplitudes. Defaults to None.
            center_mins (list, optional): List of minimum values for peak centers. Defaults to None.
            center_maxs (list, optional): List of maximum values for peak centers. Defaults to None.
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
        n_peaks = len(centers)
        # self.dcv_best_fit_stk = np.zeros((self.len_dtg_db, self.n_repl))
        # self.dcv_r2_stk = np.zeros(self.n_repl)

        # self.dcv_peaks_stk = np.zeros((self.len_dtg_db, self.n_repl, n_peaks))
        if sigmas is None:
            sigmas = [1] * n_peaks
        if amplitudes is None:
            amplitudes = [10] * n_peaks
        if center_mins is None:
            center_mins = [None] * n_peaks
        if center_maxs is None:
            center_maxs = [None] * n_peaks
        if sigma_mins is None:
            sigma_mins = [None] * n_peaks
        if sigma_maxs is None:
            sigma_maxs = [None] * n_peaks
        if amplitude_mins is None:
            amplitude_mins = [0] * n_peaks
        if amplitude_maxs is None:
            amplitude_maxs = [None] * n_peaks

        for f in range(self.n_repl):
            y = np.abs(self.dtg_db.stk(f))
            model = LinearModel(prefix="bkg_")
            params = model.make_params(intercept=0, slope=0, vary=False)

            for p in range(n_peaks):
                prefix = f"peak_{p}"
                self.dcv_peaks.append(Measure(name=prefix))
                peak_model = GaussianModel(prefix=prefix)
                pars = peak_model.make_params()
                pars[prefix + "center"].set(
                    value=centers[p], min=center_mins[p], max=center_maxs[p]
                )
                pars[prefix + "sigma"].set(value=sigmas[p], min=sigma_mins[p], max=sigma_maxs[p])
                pars[prefix + "amplitude"].set(
                    value=amplitudes[p], min=amplitude_mins[p], max=amplitude_maxs[p]
                )
                model += peak_model
                params.update(pars)

            result = model.fit(y, params=params, x=self.temp_dtg)
            self.dcv_best_fit.add(f, -result.best_fit)
            self.dcv_r2.add(f, 1 - result.residual.var() / np.var(y))
            components = result.eval_components(x=self.temp_dtg)

            for p in range(n_peaks):
                prefix = f"peak_{p}"
                if prefix in components:
                    self.dcv_peaks[p].add(f, -components[prefix])
                else:
                    self.dcv_peaks[p].add(f, 0)

        self.deconv_computed = True

    def report(
        self,
        report_type: Literal[
            "proximate", "oxidation", "oxidation_extended", "soliddist", "soliddist_extended"
        ] = "proximate",
    ) -> pd.DataFrame:
        if report_type == "proximate":
            if not self.proximate_computed:
                self.proximate_analysis()
            variables = [self.moist_ar, self.ash_db, self.vm_db, self.fc_db]
        elif report_type == "oxidation" or report_type == "oxidation_extended":
            if not self.oxidation_computed:
                self.oxidation_analysis()
            variables = [self.temp_i, self.temp_p, self.temp_b, self.s_combustion_index]
            if report_type == "oxidation_extended":
                variables += [self.dwdtemp_max, self.dwdtemp_mean]

        elif report_type == "soliddist" or report_type == "soliddist_extended":
            if not self.soliddist_computed:
                self.soliddist_analysis()
            if report_type == "soliddist":
                variables = []
                for p in self.temp_soliddist.ave():
                    variables.append(Measure(name=f"{p:0.0f} {self.temp_unit}"))
                for f in range(self.n_repl):
                    for t, dmp in enumerate(self.dmp_soliddist.stk(f)):
                        variables[t].add(f, dmp)
            elif report_type == "soliddist_extended":
                variables = [self.temp_soliddist, self.time_soliddist, self.dmp_soliddist]
        else:
            raise ValueError(f"{report_type = } is not a valid option")

        var_names = [var.name for var in variables]
        index = [f"repl_{f}" for f in range(self.n_repl)] + ["ave", "std"]
        rep_data = []

        for f in range(self.n_repl):
            rep_data.append([var.stk(f) for var in variables])
        rep_data.append([var.ave() for var in variables])
        rep_data.append([var.std() for var in variables])

        report = pd.DataFrame(data=rep_data, columns=var_names, index=index)
        report.index.name = self.name
        self.report_types_computed.append(report_type)
        self.reports[report_type] = report
        if self.auto_save_reports:
            out_path = plib.Path(self.out_path, "single_sample_reports")
            out_path.mkdir(parents=True, exist_ok=True)
            report.to_excel(plib.Path(out_path, f"{self.name}_{report_type}.xlsx"))
        return report

    def plot_tg_dtg(self, **kwargs: dict[str, Any]) -> MyFigure:
        """
        Plot the TGA data.

        """
        if not self.proximate_computed:
            self.proximate_analysis()
        out_path = plib.Path(self.out_path, "single_sample_plots")
        out_path.mkdir(parents=True, exist_ok=True)

        keys = ["height", "width", "grid", "text_font", "x_lab", "y_lab"]
        values = [
            8,
            6,
            self.plot_grid,
            self.plot_font,
            "time [min]",
            [
                f"T [{self.temp_symbol}]",
                f"{self.tg_label} (stb)",
                f"{self.tg_label} (db)",
                f"T [{self.temp_symbol}]",
                f"{self.tg_label} (db)",
                f"{self.dtg_label} (db)",
            ],
        ]
        for kwk, kwd in zip(keys, values):
            if kwk not in kwargs.keys():
                kwargs[kwk] = kwd

        mf = MyFigure(
            rows=3,
            cols=2,
            **kwargs,
        )
        # tg plot 0, 2, 4 on the left
        for f in range(self.n_repl):
            mf.axs[0].plot(
                self.time.stk(f),
                self.temp.stk(f),
                color=clrs[f],
                linestyle=lnstls[f],
                label=self.filenames[f],
            )
            mf.axs[2].plot(
                self.time.stk(f),
                self.mp_ar.stk(f),
                color=clrs[f],
                linestyle=lnstls[f],
                label=self.filenames[f],
            )
            mf.axs[4].plot(
                self.time.stk(f),
                self.mp_db.stk(f),
                color=clrs[f],
                linestyle=lnstls[f],
                label=self.filenames[f],
            )
            mf.axs[0].vlines(
                self.time.stk(f)[self.idx_moist.stk(f)],
                self.temp.stk(f)[self.idx_moist.stk(f)] - 50,
                self.temp.stk(f)[self.idx_moist.stk(f)] + 50,
                linestyle=lnstls[f],
                color=clrs[f],
            )
            mf.axs[2].vlines(
                self.time.stk(f)[self.idx_moist.stk(f)],
                self.mp_ar.stk(f)[self.idx_moist.stk(f)] - 5,
                self.mp_ar.stk(f)[self.idx_moist.stk(f)] + 5,
                linestyle=lnstls[f],
                color=clrs[f],
            )

            if self.vm_db() < 99:
                mf.axs[0].vlines(
                    self.time.stk(f)[self.idx_vm.stk(f)],
                    self.temp.stk(f)[self.idx_vm.stk(f)] - 50,
                    self.temp.stk(f)[self.idx_vm.stk(f)] + 50,
                    linestyle=lnstls[f],
                    color=clrs[f],
                )
                mf.axs[4].vlines(
                    self.time.stk(f)[self.idx_vm.stk(f)],
                    self.mp_db.stk(f)[self.idx_vm.stk(f)] - 5,
                    self.mp_db.stk(f)[self.idx_vm.stk(f)] + 5,
                    linestyle=lnstls[f],
                    color=clrs[f],
                )
            # tg plot 1, 3, 5 on the right
            mf.axs[1].plot(
                self.time_dtg.stk(f),
                self.temp_dtg,
                color=clrs[f],
                linestyle=lnstls[f],
                label=self.filenames[f],
            )
            mf.axs[3].plot(
                self.time_dtg.stk(f),
                self.mp_db_dtg.stk(f),
                color=clrs[f],
                linestyle=lnstls[f],
                label=self.filenames[f],
            )
            mf.axs[5].plot(
                self.time_dtg.stk(f),
                self.dtg_db.stk(f),
                color=clrs[f],
                linestyle=lnstls[f],
                label=self.filenames[f],
            )
            #
            if self.oxidation_computed:
                mf.axs[5].vlines(
                    self.time_dtg.stk(f)[self.temp_i_idx.stk(f)],
                    ymin=-1.5,
                    ymax=0,
                    linestyle=lnstls[f],
                    color=clrs[f],
                )
                mf.axs[5].vlines(
                    self.time_dtg.stk(f)[self.temp_p_idx.stk(f)],
                    ymin=np.min(self.dtg_db.stk(f)),
                    ymax=np.min(self.dtg_db.stk(f)) / 5,
                    linestyle=lnstls[f],
                    color=clrs[f],
                )
                mf.axs[5].vlines(
                    self.time_dtg.stk(f)[self.temp_b_idx.stk(f)],
                    ymin=-1.5,
                    ymax=0,
                    linestyle=lnstls[f],
                    color=clrs[f],
                )
        mf.save_figure(self.name + "_tg_dtg", out_path)
        return mf

    def plot_soliddist(self, **kwargs: dict[str, Any]) -> MyFigure:
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
        out_path = plib.Path(self.out_path, "single_sample_plots")
        out_path.mkdir(parents=True, exist_ok=True)

        keys = [
            "height",
            "width",
            "grid",
            "text_font",
            "x_lab",
            "y_lab",
            "yt_lab",
            "legend_loc",
        ]
        values = [
            5,
            6,
            self.plot_grid,
            self.plot_font,
            "time [min]",
            f"{self.tg_label} (db)",
            f"T [{self.temp_symbol}]",
            "center left",
        ]
        for kwk, kwd in zip(keys, values):
            if kwk not in kwargs.keys():
                kwargs[kwk] = kwd

        mf = MyFigure(
            rows=1,
            cols=1,
            twinx=True,
            **kwargs,
        )
        for f in range(self.n_repl):

            mf.axs[0].plot(
                self.time.stk(f),
                self.mp_db.stk(f),
                color=clrs[f],
                linestyle=lnstls[f],
                label=self.filenames[f],
            )
            mf.axts[0].plot(self.time.stk(f), self.temp.stk(f))

        for tm, mp, dmp in zip(self.time_soliddist(), self.loc_soliddist(), self.dmp_soliddist()):
            mf.axs[0].annotate(
                f"{dmp:0.0f}%", ha="center", va="top", xy=(tm - 10, mp + 1), fontsize=9
            )
        mf.save_figure(self.name + "_soliddist", out_path)
        return mf

    def plot_deconv(self, **kwargs: dict[str, Any]) -> MyFigure:
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
        out_path = plib.Path(self.out_path, "single_sample_plots")
        out_path.mkdir(parents=True, exist_ok=True)
        keys = [
            "height",
            "width",
            "grid",
            "text_font",
            "x_lab",
            "y_lab",
        ]
        values = [
            8,
            3.5,
            self.plot_grid,
            self.plot_font,
            f"T [{self.temp_symbol}]",
            f"{self.dtg_label} (db)",
        ]
        for kwk, kwd in zip(keys, values):
            if kwk not in kwargs.keys():
                kwargs[kwk] = kwd

        mf = MyFigure(
            rows=self.n_repl,
            cols=1,
            **kwargs,
        )

        # Plot DTG data
        for f in range(self.n_repl):
            mf.axs[f].plot(self.temp_dtg, self.dtg_db.stk(f), color="black", label="DTG")
            # Plot best fit and individual peaks
            mf.axs[f].plot(
                self.temp_dtg,
                self.dcv_best_fit.stk(f),
                label="best fit",
                color="red",
                linestyle="--",
            )
            clrs_p = clrs[:3] + clrs[5:]  # avoid using red
            for p, peak in enumerate(self.dcv_peaks):
                if peak.stk(f) is not None:
                    mf.axs[f].plot(
                        self.temp_dtg,
                        peak.stk(f),
                        label=peak.name,
                        color=clrs_p[p],
                        linestyle=lnstls[p],
                    )
            mf.axs[f].annotate(
                f"r$^2$={self.dcv_r2.stk(f):.2f}",
                xycoords="axes fraction",
                xy=(0.85, 0.96),
                size="x-small",
            )
        mf.save_figure(self.name + "_deconv", out_path)
        return mf


class Measure:
    """
    A class to collect and analyze a series of numerical data points or arrays.

    Attributes:
        _stk (dict): A dictionary to store the data points or numpy arrays with integer keys.
    """

    std_type: Literal["population", "sample"] = "population"
    if std_type == "population":
        np_ddof: int = 0
    elif std_type == "sample":
        np_ddof: int = 1

    @classmethod
    def set_std_type(cls, new_std_type: Literal["population", "sample"]):
        """"""
        cls.std_type = new_std_type
        if new_std_type == "population":
            cls.np_ddof: int = 0
        elif new_std_type == "sample":
            cls.np_ddof: int = 1

    def __init__(self, name: str | None = None):
        """
        Initializes the Measure class with an empty dictionary for _stk.
        """
        self.name = name
        self._stk: dict[int : np.ndarray | float] = {}
        self._ave: np.ndarray | float | None = None
        self._std: np.ndarray | float | None = None

    def __call__(self):
        return self.ave()

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
        elif isinstance(value, (list, tuple)):
            value = np.asarray(value)
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
            self._std = np.std(
                np.column_stack(list(self._stk.values())), axis=1, ddof=Measure.np_ddof
            )
            return self._std
        else:
            self._std = np.std(list(self._stk.values()), ddof=Measure.np_ddof)
            return self._std


# %%
# if __name__ == "__main__":


test_dir: plib.Path = plib.Path(
    r"C:\Users\mp933\OneDrive - Cornell University\Python\tga_data_analysis\tests\data"
)

# %%
# m1 = Measure()
# values = [1, 4, 7]
# for repl, value in enumerate(values):
#     m1.add(repl, value)
# assert m1.ave() == np.average(values)
# assert m1.std() == np.std(values)
# # %%
# m2 = Measure()
# values = [[1, 4, 5], [2, 6, 7], [3, 8, 9]]
# ave = [2, 6, 7]
# std = 0
# for repl, value in enumerate(values):
#     m2.add(repl, value)
# print(m2.ave())
# print(m2.std())

# %%
proj = Project(test_dir, name="test", temp_unit="K")
# %%
cell = Sample(
    project=proj,
    name="cell",
    filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
    time_moist=38,
    time_vm=None,
)
mf = cell.plot_tg_dtg(x_ticklabels_rotation=0)
# %%
#
# %%
# misc = Sample(
#     project=proj, name="misc", filenames=["MIS_1", "MIS_2", "MIS_3"], time_moist=38, time_vm=147
# )
proj = Project(test_dir, name="test", temp_unit="C")
sda = Sample(
    project=proj, name="sda", filenames=["SDa_1", "SDa_2", "SDa_3"], time_moist=38, time_vm=None
)
sdb = Sample(
    project=proj, name="sdb", filenames=["SDb_1", "SDb_2", "SDb_3"], time_moist=38, time_vm=None
)
proj.plot_multi_tg()
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
# dig = Sample(
#     project=proj, name="dig", filenames=["DIG10_1", "DIG10_2", "DIG10_3"], time_moist=22, time_vm=98
# )
# # %%
# for sample in proj.samples.values():
#     for report_type in [
#         "proximate",
#         "oxidation",
#         "oxidation_extended",
#         "soliddist",
#         "soliddist_extended",
#     ]:
#         sample.report(report_type)
#     mf = sample.plot_tg_dtg()

# # %%
# mf = sda.plot_soliddist()

# mf = sdb.plot_soliddist()
# # %%
# misc.deconv_analysis([280 + 273, 380 + 273])
# cell.deconv_analysis([310 + 273, 450 + 273, 500 + 273])
# mf = misc.plot_deconv()
# mf = cell.plot_deconv()
# # %%
# for report_type in [
#     "proximate",
#     "oxidation",
#     "oxidation_extended",
#     "soliddist",
#     # "soliddist_extended",  # not supported
# ]:
#     for report_style in ["repl_ave_std", "ave_std", "ave_pm_std"]:
#         print(f"{report_type = }, {report_style = }")
#         proj.multi_report(report_type=report_type, report_style=report_style)

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
    report_type="proximate", x_ticklabels_rotation=30, legend_loc="best", legend_bbox_xy=(1, 1.01)
)
# %%
rep = proj.plot_multireport(report_type="oxidation", x_ticklabels_rotation=0)
# %%


# %%
