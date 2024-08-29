# %%
from __future__ import annotations
from typing import Literal
import pathlib as plib
import numpy as np
from tga_data_analysis.tga import Project, Sample, Measure
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
        self.pre_exponential_factor = np.zeros(len(self.alpha))
        self.x_matrix = np.zeros((len(self.alpha), len(self.ramps)))
        self.y_matrix = np.zeros((len(self.alpha), len(self.ramps)))
        self.fits: list[np.poly1d] = []

        self.kas_analysis_computed: bool = False
        self.reaction_average_heating_rate: np.array | None = None
        self.reaction_time: np.array | None = None
        self.reaction_temp: np.array | None = None
        self.reaction_rates: np.array | None = None
        self.reaction_alphas: np.array | None = None
        self.reaction_time_of_max_rate: float | None = None
        self.reaction_temp_of_max_rate: float | None = None

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
            if not sample.dtg_computed:
                sample.dtg_analysis()
            temp = sample.temp_dtg() + c_to_k if sample.temp_unit == "C" else sample.temp_dtg()
            alpha_mass = 1 - (sample.tg_dtg() - np.min(sample.tg_dtg())) / (
                np.max(sample.tg_dtg()) - np.min(sample.tg_dtg())
            )
            for alpha_idx, alpha_val in enumerate(self.alpha):
                conversion_index = np.argmax(alpha_mass > alpha_val)
                self.x_matrix[alpha_idx, idx] = 1000 / temp[conversion_index]
                self.y_matrix[alpha_idx, idx] = np.log(
                    self.ramps[idx] / temp[conversion_index] ** 2
                )
        for a, alpha in enumerate(self.alpha):
            p, cov = np.polyfit(self.x_matrix[a, :], self.y_matrix[a, :], 1, cov=True)
            self.fits.append(np.poly1d(p))
            self.activation_energy[a] = -p[0] * r_gas_constant
            self.activation_energy_std[a] = np.sqrt(cov[0][0]) * r_gas_constant
            # Calculate pre-exponential factor A
            self.pre_exponential_factor[a] = np.exp(p[1]) * (
                self.activation_energy[a] / (r_gas_constant * alpha)
            )

        self.kas_analysis_computed = True
        return None

    def compute_rate_at_conditions(
        self,
        time_ramp_end_s: float = 1,
        time_plateaux_end_s: float | None = None,
        temp_ramp_start_kelvin: float = 298.15,
        temp_ramp_end_kelvin: float = 1645,
        temp_sigmoid_steepness: float | None = None,
        steps: int = 10000,
        reaction_mechanism: Literal["first-order", "3D-diffusion"] = "first-order",
        include_std_of_activation_energy: bool = False,
    ):
        """
        Compute the reaction rate at given conditions.
        reaction mechanisms taken from doi:10.1016/j.tca.2011.03.034

        :param time_ramp_end_s: The end time of the ramp in seconds.
        :type time_ramp_end_s: float
        :param time_plateaux_end_s: The end time of the plateaux in seconds.
                                   If None, the plateaux is not considered.
        :type time_plateaux_end_s: float or None
        :param temp_ramp_start_kelvin: The starting temperature of the ramp in Kelvin.
        :type temp_ramp_start_kelvin: float
        :param temp_ramp_end_kelvin: The ending temperature of the ramp in Kelvin.
        :type temp_ramp_end_kelvin: float
        :param temp_sigmoid_steepness: The steepness of the sigmoid function for temperature ramp.
                                       If None, a linear ramp is used.
        :type temp_sigmoid_steepness: float or None
        :param steps: The number of steps for the computation.
        :type steps: int
        :param reaction_mechanism: The reaction mechanism to use. Currently, only "first_order" is supported.
        :type reaction_mechanism: Literal["first_order"]
        :return: None
        """

        self.reaction_time_of_max_rate = Measure()
        self.reaction_temp_of_max_rate = Measure()

        if reaction_mechanism == "first-order":

            def mechanism(alpha):
                alpha = min(alpha, 1)
                return 1 - alpha

        elif reaction_mechanism == "3D-diffusion":

            def mechanism(alpha):
                alpha = min(alpha, 1)
                return 3 / 2 * (1 - alpha) ** (2 / 3) / (1 - (1 - alpha) ** (1 / 3))

        elif reaction_mechanism == "contracting-sphere":

            def mechanism(alpha):
                alpha = min(alpha, 1)
                return 3 * (1 - alpha) ** (2 / 3)

        elif reaction_mechanism == "contracting-cylinder":

            def mechanism(alpha):
                alpha = min(alpha, 1)
                return 2 * np.sqrt(1 - alpha)

        if temp_sigmoid_steepness is None:
            temps = np.linspace(temp_ramp_start_kelvin, temp_ramp_end_kelvin, steps)
        else:

            temps = sigmoid_vector(
                temp_ramp_start_kelvin, temp_ramp_end_kelvin, steps, temp_sigmoid_steepness
            )
        if time_plateaux_end_s is None:
            time = np.linspace(0, time_ramp_end_s, steps)
        else:
            if time_plateaux_end_s < time_ramp_end_s:
                # print a warning and set time plat to None and break the else
                print("WARNING: time_plateaux_end_s < time_ramp_end_s")
                time_plateaux_end_s = None
                time = np.linspace(0, time_ramp_end_s, steps)
            else:

                steps_plateaux = int(
                    steps * (time_plateaux_end_s - time_ramp_end_s) / (time_ramp_end_s)
                )
                temps_plateaux = np.ones(steps_plateaux) * temp_ramp_end_kelvin
                temps = np.concatenate((temps, temps_plateaux))
                time = np.linspace(0, time_plateaux_end_s, steps + steps_plateaux)

        dt = time[-1] / len(time)
        initial_mass = 100  # Initial mass in mg
        final_mass = 0  # Final mass in mg

        kas_activation_energy = Measure()
        reaction_rates = Measure()
        masses = Measure()
        alphas = Measure()
        if include_std_of_activation_energy:
            std_factors = [-1, 0, 1]  # -1, 0, 1 std deviations
        else:
            std_factors = [0]  # only the nominal value
        kas_alpha = self.alpha
        kas_pre_exp_factor = self.pre_exponential_factor
        for s, std_factor in enumerate(std_factors):
            masses.add(s, np.ones_like(temps) * initial_mass)
            alphas.add(s, np.zeros_like(temps))
            reaction_rates.add(s, np.zeros_like(temps))
            kas_activation_energy.add(
                s, (self.activation_energy + std_factor * np.abs(self.activation_energy_std)) * 1000
            )  # convert Ea from kJ/mol to J/mol
            for t, temp in enumerate(temps[1:]):
                kas_idx_alpha = np.argmax(kas_alpha <= alphas.stk(s)[t])

                act_energy = kas_activation_energy.stk(s)[kas_idx_alpha]
                pre_exp_factor = kas_pre_exp_factor[kas_idx_alpha]
                reaction_rates.stk(s)[t + 1] = (
                    -pre_exp_factor
                    * np.exp(-act_energy / (8.314 * temp))
                    * mechanism(alphas.stk(s)[t])
                )
                if reaction_rates.stk(s)[t + 1] > 0:
                    reaction_rates.stk(s)[t + 1] = 0
                masses.stk(s)[t + 1] = masses.stk(s)[t] + reaction_rates.stk(s)[t + 1] * dt
                alphas.stk(s)[t + 1] = (initial_mass - masses.stk(s)[t + 1]) / (
                    initial_mass - final_mass
                )
            max_rate_diff_idx = np.argmin(
                np.diff(reaction_rates.stk(s), prepend=reaction_rates.stk(s)[0])
            )
            self.reaction_time_of_max_rate.add(s, time[max_rate_diff_idx])
            self.reaction_temp_of_max_rate.add(s, temps[max_rate_diff_idx])
        # find index of the first element where the rate is 10 % of the max value
        self.reaction_average_heating_rate = (temps[-1] - temps[0]) / time[-1]  # K/s
        self.reaction_time = time
        self.reaction_temp = temps
        self.reaction_rates = reaction_rates
        self.reaction_alphas = alphas

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
        # if self.alpha is longer than the number of colors, keep only alphas at regular intervals
        if len(self.alpha) > 10:
            alpha_idx = np.arange(0, len(self.alpha), len(self.alpha) // 10)
        else:
            alpha_idx = range(len(self.alpha))
        for i, alpha_val in enumerate(self.alpha[alpha_idx]):
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
            "filename": self.name + "_activation_energy",
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

    def plot_pre_exponential_factor(
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

        out_path = plib.Path(self.out_path, "pre_exponential_plots")
        out_path.mkdir(parents=True, exist_ok=True)

        default_kwargs = {
            "filename": self.name + "_pre_exponential_factor",
            "out_path": out_path,
            "height": 4,
            "width": 4,
            "grid": self.plot_grid,
            "text_font": self.plot_font,
            "x_lab": r"$\alpha$ [-]",
            "y_lab": r"A (pre-exponential factor) [1/s]",
            "legend": None,
        }
        # Update kwargs with the default key-value pairs if the key is not present in kwargs
        kwargs = {**default_kwargs, **kwargs}
        myfig = MyFigure(rows=1, cols=1, **kwargs)

        myfig.axs[0].plot(self.alpha, self.pre_exponential_factor)

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


def plot_multi_pre_exponential_factor(
    kassamples: list[KasSample],
    labels: list[str] | None = None,
    filename: str = "plot",
    log_scale_y_axis: bool = True,
    **kwargs,
) -> MyFigure:
    """
    Plot the pre-exponential factor for multiple KAS analyses, comparing their kinetic parameters.

    This function creates a plot showing the pre-exponential factor against conversion for a series of KasSample objects,
    allowing for a comparative analysis of different samples or conditions.

    :param kassamples: A list of KasSample objects for which the pre-exponential factor plots are generated.
    :type kassamples: list[KasSample]
    :param labels: Labels corresponding to each KasSample object. If None, the names of the samples are used.
    :type labels: list[str], optional
    :param filename: The base name for the file to save the plot. Defaults to "plot".
    :type filename: str
    :param kwargs: Additional keyword arguments for plot customization.
    :type kwargs: dict
    :return: A MyFigure instance containing the comparative pre-exponential factor plot.
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
        "filename": filename + "_pre_exponential_factor",
        "out_path": out_path,
        "height": 4,
        "width": 4,
        "grid": kassamples[0].plot_grid,
        "text_font": kassamples[0].plot_font,
        "x_lab": r"$\alpha$ [-]",
        "y_lab": r"A (pre-exponential factor) [1/s]",
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
            sample.pre_exponential_factor,
            color=colors[i],
            linestyle=linestyles[i],
            label=labels[i],
        )
        if log_scale_y_axis:
            myfig.axs[0].set_yscale("log")
    myfig.save_figure()
    return myfig

    # Example data (replace with your actual data)


def plot_multi_rate_at_conditions(
    kassamples: list[KasSample],
    labels: list[str] | None = None,
    filename: str = "plot",
    time_ramp_end_s: float = 1,
    time_plateaux_end_s: float | None = 0,
    temp_ramp_start_kelvin: float = 298.15,
    temp_ramp_end_kelvin: float = 1645,
    temp_sigmoid_steepness: float | None = None,
    steps: int = 10000,
    time_units: Literal["ms", "s"] = "ms",
    reaction_mechanism: Literal["first-order", "3D-diffusion"] = "first-order",
    **kwargs,
):
    """
    Plot the reaction rate at specified conditions for multiple KasSamples.

    Parameters:
    - kassamples (list[KasSample]): List of KasSample objects.
    - labels (list[str], optional): List of labels for the samples. If None, the sample names will be used as labels. Default is None.
    - filename (str, optional): Filename for the plot. Default is "plot".
    - time_ramp_end_s (float, optional): End time for the reaction rate calculation in seconds. Default is 1.
    - time_plateaux_end_s (float, optional): Time at which the reaction reaches a plateau. Default is 0.
    - temp_ramp_start_kelvin (float, optional): Starting temperature in Kelvin. Default is 298.15.
    - temp_ramp_end_kelvin (float, optional): Ending temperature in Kelvin. Default is 1645.
    - temp_sigmoid_steepness (float, optional): Steepness of the sigmoid function used to calculate the reaction rate. Default is 0.005.
    - steps (int, optional): Number of steps used in the calculation. Default is 100000.
    - **kwargs: Additional keyword arguments for customizing the plot.

    Returns:
    - myfig (MyFigure): The generated plot as a MyFigure object.
    """

    if labels is None:
        labels = [sample.name for sample in kassamples]
    for sample in kassamples:
        if not sample.kas_analysis_computed:
            sample.kas_analysis()
        sample.compute_rate_at_conditions(
            time_ramp_end_s=time_ramp_end_s,
            time_plateaux_end_s=time_plateaux_end_s,
            temp_ramp_start_kelvin=temp_ramp_start_kelvin,
            temp_ramp_end_kelvin=temp_ramp_end_kelvin,
            temp_sigmoid_steepness=temp_sigmoid_steepness,
            reaction_mechanism=reaction_mechanism,
            steps=steps,
        )

    out_path = plib.Path(kassamples[0].out_path, "multisample_plots")
    out_path.mkdir(parents=True, exist_ok=True)

    default_kwargs = {
        "filename": filename + "_rate",
        "out_path": out_path,
        "height": 6,
        "width": 4,
        "grid": kassamples[0].plot_grid,
        "text_font": kassamples[0].plot_font,
        "x_lab": f"Time [{time_units}]",
        "yt_lab": "Temperature [K]",
        "y_lab": ["Extent of Reaction [-]", "Reaction Rate [1/s]"],
    }
    kwargs = {**default_kwargs, **kwargs}
    myfig = MyFigure(
        rows=2,
        cols=1,
        twinx=True,
        **kwargs,
    )
    time_plot = (
        kassamples[0].reaction_time * 1000 if time_units == "ms" else kassamples[0].reaction_time
    )
    myfig.axts[0].plot(
        time_plot,
        kassamples[0].reaction_temp,
        color="k",
        linestyle=linestyles[len(kassamples) + 1],
        label="T",
    )
    for k, sample in enumerate(kassamples):
        myfig.axs[0].plot(
            time_plot,
            sample.reaction_alphas.ave(),
            color=colors[k],
            linestyle=linestyles[k],
            label=sample.name,
        )
        myfig.axs[1].plot(
            time_plot,
            sample.reaction_rates.ave(),
            color=colors[k],
            linestyle=linestyles[k],
            label=sample.name,
        )
    myfig.axts[1].set_visible(False)
    # print the time and temperature of the maximum reaction rate
    print(f"Name\tHeating Rate [K/ms]\tT at max rate [K]\tTime at max rate [{time_units}]")
    for k, sample in enumerate(kassamples):
        print(
            f"{sample.name}\t{sample.reaction_average_heating_rate/1000:.0f}"
            + f"\t{sample.reaction_temp_of_max_rate.ave():.0f}"
            + f"\t{sample.reaction_time_of_max_rate.ave()*1000:.1f}"
        )

    myfig.save_figure()
    return myfig


def sigmoid_vector(t0: float, t1: float, steps: int, k: float = 0.005) -> np.ndarray:
    """
    Generate a sigmoid-shaped vector that starts from t0 and reaches t1 with a specified number of steps.

    :param t0: Start value of the vector.
    :param t1: End value of the vector.
    :param steps: Number of steps in the vector.
    :param k: Steepness of the sigmoid curve. Default is 1.
    :return: Sigmoid-shaped vector.
    """
    t = np.linspace(t0, t1, steps)
    t_m = (t0 + t1) / 2
    sigmoid = 1 / (1 + np.exp(-k * (t - t_m)))
    normalized_sigmoid = (sigmoid - np.min(sigmoid)) / (np.max(sigmoid) - np.min(sigmoid))
    scaled_sigmoid = t0 + (t1 - t0) * normalized_sigmoid

    return scaled_sigmoid


# %%
