# %%
import pathlib as plib
import numpy as np
from tga_data_analysis.tga import Project, Sample
from myfigure.myfigure import MyFigure, colors, linestyles, markers


class KasSample:
    """
    A class to handle and analyze kinetic data using the Kissinger-Akahira-Sunose (KAS) method.
    It provides functionalities to perform KAS analysis on a set of samples, plot analysis results,
    and compare different samples' kinetic parameters.
    """

    def __init__(
        self,
        project: Project,
        samples: list[Sample] | None = None,
        name: str | None = None,
        ramps: list[float] | None = None,
        alpha: list[float] | None = None,
    ):
        """
        Initialize a KasSample object with parameters for KAS analysis.

        :param project: The Project object associated with the kinetic analysis.
        :type project: Project
        :param samples: A list of Sample objects to be analyzed. If None, all samples in the project are used.
        :type samples: list[Sample], optional
        :param name: An optional name for the KasSample object, used for identification.
        :type name: str, optional
        :param ramps: A list of heating rates for each sample. If None, the heating rates are taken from the samples.
        :type ramps: list[float], optional
        :param alpha: A list of conversion values to be analyzed. If None, a default range is used.
        :type alpha: list[float], optional
        """
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
        Perform the KAS analysis across the provided samples, calculating the activation energy for different conversions.

        This method computes the activation energy for the given conversion values (alpha) using the KAS method.
        The results are stored within the object for later access and visualization.
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
        Plot the KAS analysis isolines for the set of samples.

        This method generates a plot of the KAS isolines, providing a visual representation of the kinetic analysis results.

        :param kwargs: Additional keyword arguments for plot customization.
        :type kwargs: dict
        :return: A MyFigure instance containing the isoline plot.
        :rtype: MyFigure
        """

        if not self.kas_analysis_computed:
            self.kas_analysis()

        out_path = plib.Path(self.out_path, "isolines_plots")
        out_path.mkdir(parents=True, exist_ok=True)

        default_kwargs = {
            "filename": self.name + "_isolines",
            "out_path": out_path,
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
            myfig.axs[0].plot(x_new, y_new, color=colors[i], linestyle=linestyles[i], label=lab)

            # Plotting the data points
            myfig.axs[0].plot(
                self.x_matrix[i, :],
                self.y_matrix[i, :],
                color=colors[i],
                linestyle="None",
                marker=markers[i],
            )
        myfig.save_figure()
        return myfig

    def plot_activation_energy(
        self,
        **kwargs,
    ) -> MyFigure:
        """
        Plot the activation energy as a function of conversion for the analyzed samples.

        This method generates a plot showing the variation of activation energy with conversion, offering insights into the kinetic behavior of the sample.

        :param kwargs: Additional keyword arguments for plot customization.
        :type kwargs: dict
        :return: A MyFigure instance containing the activation energy plot.
        :rtype: MyFigure
        """
        if not self.kas_analysis_computed:
            self.kas_analysis()

        out_path = plib.Path(self.out_path, "activation_energy_plots")
        out_path.mkdir(parents=True, exist_ok=True)

        default_kwargs = {
            "filename": self.name + "_isolines",
            "out_path": out_path,
            "height": 4,
            "width": 4,
            "grid": self.plot_grid,
            "text_font": self.plot_font,
            "x_lab": r"$\alpha$ [-]",
            "y_lab": r"$E_{a}$ [kJ/mol]",
            "legend": None,
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
        myfig.save_figure()
        return myfig


def plot_multi_activation_energy(
    kassamples: list[KasSample],
    labels: list[str] | None = None,
    filename: str = "plot",
    **kwargs,
) -> MyFigure:
    """
    Plot the activation energy for multiple KAS analyses, comparing their kinetic parameters.

    This function creates a plot showing the activation energy against conversion for a series of KasSample objects,
    allowing for a comparative analysis of different samples or conditions.

    :param kassamples: A list of KasSample objects for which the activation energy plots are generated.
    :type kassamples: list[KasSample]
    :param labels: Labels corresponding to each KasSample object. If None, the names of the samples are used.
    :type labels: list[str], optional
    :param filename: The base name for the file to save the plot. Defaults to "plot".
    :type filename: str
    :param kwargs: Additional keyword arguments for plot customization.
    :type kwargs: dict
    :return: A MyFigure instance containing the comparative activation energy plot.
    :rtype: MyFigure
    """

    if labels is None:
        labels = [sample.name for sample in kassamples]
    for sample in kassamples:
        if not sample.kas_analysis_computed:
            sample.kas_analysis()

    out_path = plib.Path(kassamples[0].out_path, "multisample_plots")
    out_path.mkdir(parents=True, exist_ok=True)

    default_kwargs = {
        "filename": filename + "_activation_energy",
        "out_path": out_path,
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
            color=colors[i],
            linestyle=linestyles[i],
            label=labels[i],
        )

        # Plotting the data points
        myfig.axs[0].fill_between(
            sample.alpha,
            sample.activation_energy - sample.activation_energy_std,
            sample.activation_energy + sample.activation_energy_std,
            alpha=0.3,
            color=colors[i],
            linestyle=linestyles[i],
        )
    myfig.save_figure()
    return myfig
