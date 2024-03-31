# %%
import numpy as np
import pathlib as plib
from tga_data_analysis.tga import Project, Sample
from tga_data_analysis.myfigure import MyFigure, clrs, lttrs, lnstls, mrkrs


class KasSample:

    def __init__(
        self,
        project: Project,
        samples: list[Sample] | None = None,
        name: str | None = None,
        ramps: list[float] | None = None,
        alpha: list[float] | None = None,
    ):
        self.plot_grid = project.plot_grid
        self.plot_font = project.plot_font
        self.out_path = plib.Path(project.out_path, "kas_analysis")
        if samples is None:
            self.samples = project.samples
        else:
            self.samples = samples
        self.samplenames = [sample.name for sample in self.samples]
        if name is None:
            self.name = "".join(
                [
                    char
                    for char in self.samplenames[0]
                    if all(char in samplename for samplename in self.samplenames)
                ]
            )
        else:
            self.name = name

        if ramps is None:
            self.ramps = [sample.heating_rate_deg_min for sample in self.samples]
        else:
            self.ramps = ramps
        if alpha is None:
            self.alpha = np.arange(0.1, 0.85, 0.05)
        else:
            self.alpha = alpha

        self.activation_energy = np.zeros(len(self.alpha))
        self.activation_energy_std = np.zeros(len(self.alpha))
        self.x_matrix = np.zeros((len(self.alpha), len(self.ramps)))
        self.y_matrix = np.zeros((len(self.alpha), len(self.ramps)))
        self.fits: list[np.poly1d] = []

        self.kas_analysis_computed: bool = False

    def kas_analysis(
        self,
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

        r_gas_constant = 8.314462618  # Universal gas constant in J/(mol*K)
        c_to_k = 273.15

        for idx, sample in enumerate(self.samples):
            temp = sample.temp_dtg + c_to_k if sample.temp_unit == "C" else sample.temp_dtg
            alpha_mass = 1 - (sample.mp_db_dtg() - np.min(sample.mp_db_dtg())) / (
                np.max(sample.mp_db_dtg()) - np.min(sample.mp_db_dtg())
            )
            for alpha_idx, alpha_val in enumerate(self.alpha):
                conversion_index = np.argmax(alpha_mass > alpha_val)
                self.x_matrix[alpha_idx, idx] = 1000 / temp[conversion_index]
                self.y_matrix[alpha_idx, idx] = np.log(
                    self.ramps[idx] / temp[conversion_index] ** 2
                )
        for i in range(len(self.alpha)):
            p, cov = np.polyfit(self.x_matrix[i, :], self.y_matrix[i, :], 1, cov=True)
            self.fits.append(np.poly1d(p))
            self.activation_energy[i] = -p[0] * r_gas_constant
            self.activation_energy_std[i] = np.sqrt(cov[0][0]) * r_gas_constant

        self.kas_analysis_computed = True
        return None

    def plot_isolines(
        self,
        **kwargs,
    ) -> MyFigure:
        """
        Plot isolines for KAS analysis.

        Parameters:
        - exps (list): List of experiments.
        - kas_names (list, optional): List of names for each KAS analysis. If not provided, the names will be extracted from the KAS analysis data.
        - filename (str, optional): Name of the output file. Default is 'KAsIso'.
        - paper_col (float, optional): Width of the plot in inches. Default is 0.78.
        - hgt_mltp (float, optional): Height multiplier for the plot. Default is 1.25.
        - x_lim (tuple, optional): Limits for the x-axis. Default is None.
        - y_lim (tuple, optional): Limits for the y-axis. Default is None.
        - annt_names (bool, optional): Whether to annotate the names of the KAS analysis. Default is True.
        - annotate_lttrs (bool, optional): Whether to annotate the letters for each KAS analysis. Default is False.
        - leg_cols (int, optional): Number of columns in the legend. Default is 1.
        - bboxtoanchor (bool, optional): Whether to place the legend outside of the plot area. Default is True.
        - x_anchor (float, optional): X-coordinate for the legend anchor. Default is 1.13.
        - y_anchor (float, optional): Y-coordinate for the legend anchor. Default is 1.02.
        - legend_loc (str, optional): Location of the legend. Default is 'best'.
        """

        if not self.kas_analysis_computed:
            self.kas_analysis()

        out_path = plib.Path(self.out_path, "isolines_plots")
        out_path.mkdir(parents=True, exist_ok=True)

        default_kwargs = {
            "height": 4,
            "width": 5,
            "grid": self.plot_grid,
            "text_font": self.plot_font,
            "x_lab": "1000/T [1/K]",
            "y_lab": r"ln($\beta$/T$^{2}$)",
        }
        # Update kwargs with the default key-value pairs if the key is not present in kwargs
        kwargs = {**default_kwargs, **kwargs}
        myfig = MyFigure(rows=1, cols=1, **kwargs)

        for i, alpha_val in enumerate(self.alpha):
            lab = rf"$\alpha$={alpha_val:0.2f}"
            # Assuming self.fits[i] is the fit for the i-th alpha
            fit_fn = self.fits[i]
            x_new = np.linspace(np.min(self.x_matrix[i, :]), np.max(self.x_matrix[i, :]), 100)
            y_new = fit_fn(x_new)

            # Plotting the fit line
            myfig.axs[0].plot(x_new, y_new, color=clrs[i], linestyle=lnstls[i], label=lab)

            # Plotting the data points
            myfig.axs[0].plot(
                self.x_matrix[i, :],
                self.y_matrix[i, :],
                color=clrs[i],
                linestyle="None",
                marker=mrkrs[i],
            )
        myfig.save_figure(self.name + "_isolines", out_path)
        return myfig

    def plot_activation_energy(
        self,
        **kwargs,
    ) -> MyFigure:
        """
        Plot isolines for KAS analysis.

        Parameters:
        - exps (list): List of experiments.
        - kas_names (list, optional): List of names for each KAS analysis. If not provided, the names will be extracted from the KAS analysis data.
        - filename (str, optional): Name of the output file. Default is 'KAsIso'.
        - paper_col (float, optional): Width of the plot in inches. Default is 0.78.
        - hgt_mltp (float, optional): Height multiplier for the plot. Default is 1.25.
        - x_lim (tuple, optional): Limits for the x-axis. Default is None.
        - y_lim (tuple, optional): Limits for the y-axis. Default is None.
        - annt_names (bool, optional): Whether to annotate the names of the KAS analysis. Default is True.
        - annotate_lttrs (bool, optional): Whether to annotate the letters for each KAS analysis. Default is False.
        - leg_cols (int, optional): Number of columns in the legend. Default is 1.
        - bboxtoanchor (bool, optional): Whether to place the legend outside of the plot area. Default is True.
        - x_anchor (float, optional): X-coordinate for the legend anchor. Default is 1.13.
        - y_anchor (float, optional): Y-coordinate for the legend anchor. Default is 1.02.
        - legend_loc (str, optional): Location of the legend. Default is 'best'.
        """

        if not self.kas_analysis_computed:
            self.kas_analysis()

        out_path = plib.Path(self.out_path, "activation_energy_plots")
        out_path.mkdir(parents=True, exist_ok=True)

        default_kwargs = {
            "height": 4,
            "width": 4,
            "grid": self.plot_grid,
            "text_font": self.plot_font,
            "x_lab": r"$\alpha$ [-]",
            "y_lab": r"$E_{a}$ [kJ/mol]",
        }
        # Update kwargs with the default key-value pairs if the key is not present in kwargs
        kwargs = {**default_kwargs, **kwargs}
        myfig = MyFigure(rows=1, cols=1, **kwargs)

        myfig.axs[0].plot(self.alpha, self.activation_energy)

        # Plotting the data points
        myfig.axs[0].fill_between(
            self.alpha,
            self.activation_energy - self.activation_energy_std,
            self.activation_energy + self.activation_energy_std,
            alpha=0.3,
        )
        myfig.save_figure(self.name + "_isolines", out_path)
        return myfig


def plot_multi_activation_energy(
    kassamples: list[KasSample],
    labels: list[str] | None = None,
    filename: str = "plot",
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

    if labels is None:
        labels = [sample.name for sample in kassamples]
    for sample in kassamples:
        if not sample.kas_analysis_computed:
            sample.kas_analysis()

    out_path = plib.Path(kassamples[0].out_path, "multisample_plots")
    out_path.mkdir(parents=True, exist_ok=True)

    default_kwargs = {
        "height": 4,
        "width": 4,
        "grid": kassamples[0].plot_grid,
        "text_font": kassamples[0].plot_font,
        "x_lab": r"$\alpha$ [-]",
        "y_lab": r"$E_{a}$ [kJ/mol]",
    }
    kwargs = {**default_kwargs, **kwargs}
    myfig = MyFigure(
        rows=1,
        cols=1,
        **kwargs,
    )
    for i, sample in enumerate(kassamples):
        myfig.axs[0].plot(
            sample.alpha,
            sample.activation_energy,
            color=clrs[i],
            linestyle=lnstls[i],
            label=labels[i],
        )

        # Plotting the data points
        myfig.axs[0].fill_between(
            sample.alpha,
            sample.activation_energy - sample.activation_energy_std,
            sample.activation_energy + sample.activation_energy_std,
            alpha=0.3,
            color=clrs[i],
            linestyle=lnstls[i],
        )
    myfig.save_figure(filename + "_activation_energy", out_path)
    return myfig


# %%
test_dir: plib.Path = plib.Path(
    r"C:\Users\mp933\OneDrive - Cornell University\Python\tga_data_analysis\tests\data"
)
proj = Project(test_dir, name="test", temp_unit="K")
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
cell_ox50 = Sample(
    project=proj,
    name="cell_ox50",
    load_skiprows=8,
    filenames=["CLSOx50_4", "CLSOx50_5", "CLSOx50_6"],
    time_moist=38,
    time_vm=None,
    heating_rate_deg_min=50,
)
cell_ox100 = Sample(
    project=proj,
    name="cell_ox100",
    load_skiprows=8,
    filenames=["CLSOx100_4", "CLSOx100_5", "CLSOx100_6"],
    time_moist=38,
    time_vm=None,
    heating_rate_deg_min=100,
)
# %%
cell = KasSample(proj, samples=[cell_ox5, cell_ox10, cell_ox50, cell_ox100], name="cellulose")
cell.plot_isolines(legend_bbox_xy=(1, 1))
cell.plot_activation_energy(legend_bbox_xy=(1, 1))
# %%
pc_ox10 = Sample(
    project=proj,
    name="pc_ox10",
    load_skiprows=8,
    filenames=["PCOx10_1"],
    time_moist=38,
    time_vm=None,
    heating_rate_deg_min=10,
)
pc_ox50 = Sample(
    project=proj,
    name="pc_ox50",
    load_skiprows=8,
    filenames=["PCOx50_1"],
    time_moist=38,
    time_vm=None,
    heating_rate_deg_min=50,
)
pc_ox100 = Sample(
    project=proj,
    name="pc_ox100",
    load_skiprows=8,
    filenames=["PCOx100_1"],
    time_moist=38,
    time_vm=None,
    heating_rate_deg_min=100,
)
# %%
pc = KasSample(proj, samples=[pc_ox10, pc_ox50, pc_ox100], name="primary")
pc.plot_isolines(legend_bbox_xy=(1, 1))
pc.plot_activation_energy(legend_bbox_xy=(1, 1))
# %%
plot_multi_activation_energy([cell, pc])
# %%
