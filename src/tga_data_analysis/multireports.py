from __future__ import annotations
import matplotlib.pyplot as plt
import pathlib as plib
import numpy as np
import pandas as pd

from .tgaexp import TGAExp
from .plotting import lttrs, lnstls, clrs, mrkrs, figure_create, figure_save
from .myfigure import MyFigure


# this should be single


def KAS_plot_isolines(
    exp: TGAExp,
    filename: str = "plot",
    label: list[str] | None = None,
    x_lab="1000/T [1/K]",
    y_lab=r"ln($\beta$/T$^{2}$)",
    annt_names: bool = True,
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

    if not exp.KAS_computed:
        raise ValueError(f"{exp.name=} missing KAS analysis")
    out_path = plib.Path(exp.out_path, "kas_multiplots")
    out_path.mkdir(parents=True, exist_ok=True)
    xmatr = exp.kas["xmatr"]
    # for plots
    if label is None:
        label = exp.kas["name"]

    alpha = exp.kas["alpha"]
    n_alpha = len(alpha)

    x = np.linspace(
        np.min(xmatr),
        np.max(xmatr),
        100,
    )

    myfig = MyFigure(
        rows=1,
        cols=1,
        text_font=TGAExp.plot_font,
        y_lab=y_lab,
        grid=TGAExp.plot_grid,
        **kwargs,
    )
    ymaxiso = np.max(exp.kas["ymatr"])
    yminiso = np.min(exp.kas["ymatr"])
    for i in range(n_alpha):
        lab = rf"$\alpha$={alpha[i]:0.2f}"
        xmin = np.argwhere(exp.kas["v_fit"][i](x) < ymaxiso)[0][0]
        try:
            xmax = np.argwhere(exp.kas["v_fit"][i](x) < yminiso)[0][0]
        except IndexError:
            xmax = 0
        newx = x[xmin:xmax]
        myfig.axs[0].plot(newx, exp.kas["v_fit"][i](newx), color=clrs[i], linestyle=lnstls[i])
        myfig.axs[0].plot(
            exp.kas["xmatr"][i, :],
            exp.kas["ymatr"][i, :],
            color=clrs[i],
            linestyle="None",
            marker=mrkrs[i],
        )
        myfig.axs[0].plot([], [], color=clrs[i], linestyle=lnstls[i], marker=mrkrs[i], label=lab)
    myfig.save_figure(filename, out_path)
    return myfig


def KAS_plot_Ea(
    exps: list[TGAExp],
    filename: str = "plot",
    labels: list[str] | None = None,
    x_lab=r"$\alpha$ [-]",
    y_lab=r"$E_{a}$ [kJ/mol]",
    plot_type="scatter",
    **kwargs,
) -> MyFigure:
    """
    Plot the activation energy (Ea) for multiple experiments.

    Parameters:
    - exps (list): List of experiments.
    - kas_names (list, optional): List of names for the KAS analysis. If not provided, the names will be extracted from the experiments.
    - filename (str, optional): Name of the output file. Default is 'KASEa'.
    - paper_col (float, optional): Color of the plot background. Default is 0.78.
    - hgt_mltp (float, optional): Height multiplier for the plot. Default is 1.25.
    - x_lim (list, optional): Limits of the x-axis. Default is [0.1, 0.8].
    - y_lim (list, optional): Limits of the y-axis. Default is [0, 300].
    - y_ticks (list, optional): Custom y-axis tick locations. Default is None.
    - annt_names (bool, optional): Whether to annotate the names of the experiments. Default is True.
    - annotate_lttrs (bool, optional): Whether to annotate the letters of the experiments. Default is False.
    - leg_cols (int, optional): Number of columns in the legend. Default is 1.
    - bboxtoanchor (bool, optional): Whether to place the legend outside of the plot area. Default is True.
    - x_anchor (float, optional): X-coordinate of the legend anchor. Default is 1.13.
    - y_anchor (float, optional): Y-coordinate of the legend anchor. Default is 2.02.
    - plot_type (str, optional): Type of plot. Can be 'scatter' or 'line'. Default is 'scatter'.
    - legend_loc (str, optional): Location of the legend. Default is 'best'.
    """

    for exp in exps:
        if not exp.KAS_computed:
            exp.KAS_analysis()
    out_path = plib.Path(exps[0].out_path, "KAS")
    out_path.mkdir(parents=True, exist_ok=True)
    kass = [exp.kas for exp in exps]
    if labels is None:
        labels = [kas["name"] for kas in kass]
    alphas = [kas["alpha"] for kas in kass]
    if not all(np.array_equal(alphas[0], a) for a in alphas):
        print("samples have been analyzed at different alphas")
    else:
        alpha = alphas[0]
    print(labels)
    # plot activation energy
    myfig = MyFigure(
        rows=1,
        cols=1,
        x_lab=x_lab,
        y_lab=y_lab,
        text_font=TGAExp.plot_font,
        grid=TGAExp.plot_grid,
        **kwargs,
    )
    for k, kas in enumerate(kass):
        if plot_type == "scatter":
            myfig.axs[0].errorbar(
                alpha,
                kas["Ea"],
                kas["Ea_std"],
                color="k",
                linestyle="None",
                capsize=3,
                ecolor=clrs[k],
            )
            myfig.axs[0].plot(
                alpha,
                kas["Ea"],
                color=clrs[k],
                linestyle="None",
                marker=mrkrs[k],
                label=labels[k],
            )
        elif plot_type == "line":
            myfig.axs[0].plot(alpha, kas["Ea"], color=clrs[k], linestyle=lnstls[k], label=labels[k])
            myfig.axs[0].fill_between(
                alpha,
                kas["Ea"] - kas["Ea_std"],
                kas["Ea"] + kas["Ea_std"],
                color=clrs[k],
                alpha=0.3,
            )
    myfig.save_figure(filename, out_path)
    return myfig


def oxidation_multi_plot(
    exps: list[TGAExp],
    filename: str = "plot",
    labels: list[str] | None = None,
    y_lab="T [" + TGAExp.T_symbol + "]",
    yt_lab="S (combustion index) [-]",
    x_labels_rotation: int = 0,
    **kwargs,
) -> MyFigure:
    """
    Generate a multi-plot for oxidation analysis.

    Parameters:
    - exps (list): List of experiments to be plotted.
    - filename (str): Name of the output file (default: "Oxidations").
    - smpl_labs (list): List of sample labels (default: None).
    - xlab_rot (int): Rotation angle of x-axis labels (default: 0).
    - paper_col (float): Color of the plot background (default: 0.8).
    - hgt_mltp (float): Height multiplier of the plot (default: 1.5).
    - bboxtoanchor (bool): Whether to place the legend outside the plot area (default: True).
    - x_anchor (float): X-coordinate of the legend anchor point (default: 1.13).
    - y_anchor (float): Y-coordinate of the legend anchor point (default: 1.02).
    - legend_loc (str): Location of the legend (default: 'best').
    - y_lim (tuple): Limits of the y-axis (default: None).
    - yt_lim (tuple): Limits of the twin y-axis (default: None).
    - y_ticks (list): Custom tick positions for the y-axis (default: None).
    - yt_ticks (list): Custom tick positions for the twin y-axis (default: None).

    Returns:
    - None
    """

    for exp in exps:
        if not exp.oxidation_computed:
            exp.oxidation_analysis()
    out_path = plib.Path(exps[0].out_path, "oxidation_multiplots")
    out_path.mkdir(parents=True, exist_ok=True)
    vars_bar = ["T$_i$", "T$_p$", "T$_b$"]
    vars_scat = "S"
    if not labels:  # try with labels and use name if no label is given
        labels = [exp.label if exp.label else exp.name for exp in exps]
    df_ave = pd.DataFrame(columns=vars_bar, index=labels)
    df_std = pd.DataFrame(columns=vars_bar, index=labels)
    df_ave["T$_i$"] = [exp.Ti for exp in exps]
    df_ave["T$_p$"] = [exp.Tp for exp in exps]
    df_ave["T$_b$"] = [exp.Tb for exp in exps]
    df_std["T$_i$"] = [exp.Ti_std for exp in exps]
    df_std["T$_p$"] = [exp.Tp_std for exp in exps]
    df_std["T$_b$"] = [exp.Tb_std for exp in exps]

    S_combs = [exp.S for exp in exps]
    S_combs_std = [exp.S_std for exp in exps]
    _ = kwargs.pop("x_labels_rotation", None)
    myfig = MyFigure(
        rows=1,
        cols=1,
        twinx=True,
        text_font=TGAExp.plot_font,
        y_lab=y_lab,
        yt_lab=yt_lab,
        grid=TGAExp.plot_grid,
        **kwargs,
    )
    df_ave.plot(
        kind="bar",
        ax=myfig.axs[0],
        yerr=df_std,
        capsize=2,
        width=0.85,
        ecolor="k",
        edgecolor="black",
        # rot=xlab_rot,
    )
    bars = myfig.axs[0].patches
    patterns = [None, "//", "..."]  # set hatch patterns in the correct order
    hatches = []  # list for hatches in the order of the bars
    for h in patterns:  # loop over patterns to create bar-ordered hatches
        for i in range(int(len(bars) / len(patterns))):
            hatches.append(h)
    # loop over bars and hatches to set hatches in correct order
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    myfig.axts[0].errorbar(
        x=df_ave.index,
        y=S_combs,
        yerr=S_combs_std,
        linestyle="None",
        marker=mrkrs[0],
        ecolor="k",
        capsize=2,
        markeredgecolor="k",
        color=clrs[3],
        markersize=10,
        label=vars_scat,
    )
    if x_labels_rotation == 0:
        myfig.axs[0].set_xticklabels(df_ave.index, rotation=x_labels_rotation)
    else:
        myfig.axs[0].set_xticklabels(
            df_ave.index, rotation=x_labels_rotation, ha="right", rotation_mode="anchor"
        )
    myfig.save_figure(filename, out_path)
    return myfig


def proximate_multi_plot(
    exps: list[TGAExp],
    filename: str = "plot",
    labels: list[str] | None = None,
    y_lab="mass fraction [wt%]",
    yt_lab="mean TG deviation [%]",
    x_labels_rotation: int = 0,
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
    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    out_path = plib.Path(exps[0].out_path, "proximate_multiplots")
    out_path.mkdir(parents=True, exist_ok=True)
    vars_bar = ["moisture (stb)", "VM (db)", "FC (db)", "ash (db)"]
    vars_scat = "mean TG dev."
    if not labels:  # try with labels and use name if no label is given
        labels = [exp.label if exp.label else exp.name for exp in exps]
    df_ave = pd.DataFrame(columns=vars_bar, index=labels)
    df_std = pd.DataFrame(columns=vars_bar, index=labels)
    df_ave["moisture (stb)"] = [exp.moist_ar for exp in exps]
    df_ave["VM (db)"] = [exp.vm_db for exp in exps]
    df_ave["FC (db)"] = [exp.fc_db for exp in exps]
    df_ave["ash (db)"] = [exp.ash_db for exp in exps]
    df_std["moisture (stb)"] = [exp.moist_ar_std for exp in exps]
    df_std["VM (db)"] = [exp.vm_db_std for exp in exps]
    df_std["FC (db)"] = [exp.fc_db_std for exp in exps]
    df_std["ash (db)"] = [exp.ash_db_std for exp in exps]

    aveTG = [exp.AveTGstd_p for exp in exps]
    # remove x_labels_rotation form kwargs as it is dealt with in this fucntion
    _ = kwargs.pop("x_labels_rotation", None)
    myfig = MyFigure(
        rows=1,
        cols=1,
        twinx=True,
        text_font=TGAExp.plot_font,
        y_lab=y_lab,
        yt_lab=yt_lab,
        grid=TGAExp.plot_grid,
        **kwargs,
    )
    df_ave.plot(
        ax=myfig.axs[0],
        kind="bar",
        yerr=df_std,
        capsize=2,
        width=0.85,
        ecolor="k",
        edgecolor="black",
        # rot=xlab_rot,
    )
    bars = myfig.axs[0].patches
    patterns = [None, "//", "...", "--"]  # set hatch patterns in correct order
    hatches = []  # list for hatches in the order of the bars
    for h in patterns:  # loop over patterns to create bar-ordered hatches
        for i in range(int(len(bars) / len(patterns))):
            hatches.append(h)
    # loop over bars and hatches to set hatches in correct order
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    myfig.axts[0].errorbar(
        x=df_ave.index,
        y=aveTG,
        linestyle="None",
        marker=mrkrs[0],
        color=clrs[4],
        markersize=10,
        markeredgecolor="k",
        label=vars_scat,
    )
    if x_labels_rotation == 0:
        myfig.axs[0].set_xticklabels(df_ave.index, rotation=x_labels_rotation)
    else:
        myfig.axs[0].set_xticklabels(
            df_ave.index, rotation=x_labels_rotation, ha="right", rotation_mode="anchor"
        )
    myfig.save_figure(filename, out_path)
    return myfig


def tg_multi_plot(
    exps: list[TGAExp],
    filename: str = "plot",
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

    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    out_path = plib.Path(exps[0].out_path, "tg_multiplots")
    out_path.mkdir(parents=True, exist_ok=True)
    if not labels:  # try with labels and use name if no label is given
        labels = [exp.label if exp.label else exp.name for exp in exps]

    myfig = MyFigure(
        rows=1,
        cols=1,
        text_font=TGAExp.plot_font,
        y_lab=TGAExp.TG_lab,
        x_lab="T [" + TGAExp.T_symbol + "]",
        grid=TGAExp.plot_grid,
        **kwargs,
    )
    for i, exp in enumerate(exps):
        myfig.axs[0].plot(
            exp.T,
            exp.mp_db,
            color=clrs[i],
            linestyle=lnstls[i],
            label=labels[i],
        )
        myfig.axs[0].fill_between(
            exp.T,
            exp.mp_db - exp.mp_db_std,
            exp.mp_db + exp.mp_db_std,
            color=clrs[i],
            alpha=0.3,
        )
    myfig.save_figure(filename, out_path)
    return myfig


def dtg_multi_plot(
    exps: list[TGAExp],
    filename: str = "plot",
    labels: list[str] | None = None,
    **kwargs,
) -> MyFigure:
    """
    Plot multiple DTG curves for a list of experiments.

    Args:
        exps (list): List of TGAExp objects representing the experiments.
        filename (str, optional): Name of the output file. Defaults to 'Fig'.
        paper_col (float, optional): Color of the plot background. Defaults to 0.78.
        hgt_mltp (float, optional): Height multiplier for the plot. Defaults to 1.25.
        x_lim (tuple, optional): Limits of the x-axis. Defaults to None.
        y_lim (tuple, optional): Limits of the y-axis. Defaults to None.
        y_ticks (list, optional): Custom y-axis tick labels. Defaults to None.
        lttrs (bool, optional): Whether to annotate letters on the plot. Defaults to False.
        plt_gc (bool, optional): Whether to plot the GC-MS maximum temperature line. Defaults to False.
        gc_Tlim (int, optional): Maximum temperature for the GC-MS line. Defaults to 300.
        save_as_pdf (bool, optional): Whether to save the plot as a PDF file. Defaults to False.
        save_as_svg (bool, optional): Whether to save the plot as an SVG file. Defaults to False.

    Returns:
        None
    """

    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    out_path = plib.Path(exps[0].out_path, "dtg_multiplots")
    out_path.mkdir(parents=True, exist_ok=True)
    if not labels:  # try with labels and use name if no label is given
        labels = [exp.label if exp.label else exp.name for exp in exps]
    myfig = MyFigure(
        rows=1,
        cols=1,
        text_font=TGAExp.plot_font,
        y_lab=TGAExp.DTG_lab,
        x_lab="T [" + TGAExp.T_symbol + "]",
        grid=TGAExp.plot_grid,
        **kwargs,
    )
    for i, exp in enumerate(exps):
        myfig.axs[0].plot(
            exp.T_dtg, exp.dtg_db, color=clrs[i], linestyle=lnstls[i], label=labels[i]
        )
        myfig.axs[0].fill_between(
            exp.T_dtg,
            exp.dtg_db - exp.dtg_db_std,
            exp.dtg_db + exp.dtg_db_std,
            color=clrs[i],
            alpha=0.3,
        )
    myfig.save_figure(filename, out_path)
    return myfig


def soliddist_multi_plot(
    exps: list[TGAExp],
    filename: str = "plot",
    labels: list[str] | None = None,
    annotate_percentages: bool = True,
    overlap_temperature: bool = False,
    **kwargs,
):
    """
    Plot the solid distribution for multiple experiments.

    Args:
        exps (list): List of Experiment objects.
        filename (str, optional): Name of the output file. Defaults to "Dist".
        hgt_mltp (float, optional): Height multiplier for the plot. Defaults to 1.25.
        paper_col (float, optional): Color of the plot background. Defaults to .78.
        labels (list, optional): List of labels for the experiments. Defaults to None.
        lttrs (bool, optional): Whether to annotate letters on the plot. Defaults to False.
        x_lim (list, optional): Limits for the x-axis. Defaults to None.
        y_lim (list, optional): Limits for the y-axis. Defaults to [[0, 1000], [0, 100]].
        y_ticks (list, optional): Custom tick locations for the y-axis. Defaults to None.
        print_dfs (bool, optional): Whether to print the dataframes. Defaults to True.
    """

    for exp in exps:
        if not exp.soliddist_computed:
            exp.soliddist_analysis()
    out_path = plib.Path(exps[0].out_path, "soliddist_multiplots")
    out_path.mkdir(parents=True, exist_ok=True)
    if not labels:  # try with labels and use name if no label is given
        labels = [exp.label if exp.label else exp.name for exp in exps]
    if not overlap_temperature:
        myfig = MyFigure(
            rows=2,
            x_lab="time [min]",
            y_lab=["T [" + TGAExp.T_symbol + "]", TGAExp.TG_lab + "(db)"],
            text_font=TGAExp.plot_font,
            grid=TGAExp.plot_grid,
            **kwargs,
        )
        for i, exp in enumerate(exps):
            myfig.axs[0].plot(exp.time, exp.T, color=clrs[i], linestyle=lnstls[i], label=labels[i])
            myfig.axs[0].fill_between(
                exp.time, exp.T - exp.T_std, exp.T + exp.T_std, color=clrs[i], alpha=0.3
            )
            myfig.axs[1].plot(
                exp.time, exp.mp_db, color=clrs[i], linestyle=lnstls[i], label=labels[i]
            )
            myfig.axs[1].fill_between(
                exp.time,
                exp.mp_db - exp.mp_db_std,
                exp.mp_db + exp.mp_db_std,
                color=clrs[i],
                alpha=0.3,
            )
            if annotate_percentages:

                for tm, mp, dmp in zip(exp.time_dist, exp.loc_dist, exp.dmp_dist):
                    if dmp >= 2:
                        myfig.axs[1].annotate(
                            f"{dmp:.0f}%",
                            ha="center",
                            va="top",
                            xy=(tm - 10, mp + 1),
                            fontsize=9,
                            color=clrs[i],
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.5),
                        )
        myfig.save_figure(filename, out_path)

    elif overlap_temperature:
        myfig = MyFigure(
            rows=1,
            twinx=True,
            x_lab="time [min]",
            y_lab=TGAExp.TG_lab + "(db)",
            yt_lab="T [" + TGAExp.T_symbol + "]",
            text_font=TGAExp.plot_font,
            grid=TGAExp.plot_grid,
            **kwargs,
        )
        myfig.axts[0].plot(exps[0].time, exps[0].T, color="k", linestyle=lnstls[1], label="T")
        for i, exp in enumerate(exps):

            myfig.axs[0].plot(
                exp.time, exp.mp_db, color=clrs[i], linestyle=lnstls[i], label=labels[i]
            )
            myfig.axs[0].fill_between(
                exp.time,
                exp.mp_db - exp.mp_db_std,
                exp.mp_db + exp.mp_db_std,
                color=clrs[i],
                alpha=0.3,
            )
        myfig.save_figure(filename, out_path)

    return myfig


# =============================================================================
# # functions to print reports with ave and std of multiple samples
# =============================================================================
def proximate_multi_report(exps: list[TGAExp], filename: str = "Rep") -> pd.DataFrame:
    """
    Generate a multi-sample proximate report.

    Args:
        exps (list): List of experiments.
        filename (str, optional): Name of the output file. Defaults to 'Rep'.

    Returns:
        pandas.DataFrame: DataFrame containing the multi-sample proximate report.
    """

    for exp in exps:
        if not exp.proximate_report_computed:
            exp.proximate_report()
    out_path = plib.Path(exps[0].out_path, "proximate_multireports")
    out_path.mkdir(parents=True, exist_ok=True)

    rep: pd.DataFrame = pd.DataFrame(columns=list(exps[0].proximate))
    for exp in exps:
        rep.loc[exp.label + "_ave"] = exp.proximate.loc["ave", :]
    for exp in exps:
        rep.loc[exp.label + "_std"] = exp.proximate.loc["std", :]
    rep.to_excel(plib.Path(out_path, filename + "_prox.xlsx"))
    return rep


def oxidation_multi_report(exps: list[TGAExp], filename: str = "Rep") -> pd.DataFrame:
    """
    Generate a multi-sample oxidation report.

    Args:
        exps (list): List of experiments.
        filename (str, optional): Name of the output file. Defaults to 'Rep'.

    Returns:
        pandas.DataFrame: DataFrame containing the multi-sample oxidation report.
    """
    for exp in exps:
        if not exp.oxidation_report_computed:
            exp.oxidation_report()
    out_path = plib.Path(exps[0].out_path, "oxidation_multireports")
    out_path.mkdir(parents=True, exist_ok=True)

    rep: pd.DataFrame = pd.DataFrame(columns=list(exps[0].oxidation))
    for exp in exps:
        rep.loc[exp.label + "_ave"] = exp.oxidation.loc["ave", :]
    for exp in exps:
        rep.loc[exp.label + "_std"] = exp.oxidation.loc["std", :]
    rep.to_excel(plib.Path(out_path, filename + "_oxid.xlsx"))
    return rep


def soliddist_multi_report(exps: list[TGAExp], filename: str = "Rep") -> pd.DataFrame:
    """
    Generate a multi-sample solid distance report.

    Args:
        exps (list): List of experiments.
        filename (str, optional): Name of the output file. Defaults to 'Rep'.

    Returns:
        pandas.DataFrame: DataFrame containing the solid distance report.
    """

    for exp in exps:
        if not exp.soliddist_report_computed:
            exp.soliddist_report()
    out_path = plib.Path(exps[0].out_path, "soliddist_multireports")
    out_path.mkdir(parents=True, exist_ok=True)

    rep: pd.DataFrame = pd.DataFrame(columns=list(exps[0].soliddist))
    for exp in exps:
        rep.loc[exp.label + "_ave"] = exp.soliddist.loc["ave", :]
    for exp in exps:
        rep.loc[exp.label + "_std"] = exp.soliddist.loc["std", :]
    rep.to_excel(plib.Path(out_path, filename + "_soliddist.xlsx"))
    return rep


# =============================================================================
# # functions for plotting ave and std of multiple samples
# =============================================================================
def tg_multi_plotOLD(
    exps: list[TGAExp],
    filename: str = "Fig",
    paper_col: float = 0.78,
    hgt_mltp: float = 1.25,
    x_lim=None,
    y_lim=None,
    x_ticks=None,
    y_ticks=None,
    annotate_lttrs=False,
    save_as_pdf=False,
    save_as_svg=False,
    legend_loc="upper right",
):
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
    if y_lim is None:
        y_lim = [0, 100]

    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    out_path_TGs = plib.Path(exps[0].out_path, "MultiSamplePlots")
    out_path_TGs.mkdir(parents=True, exist_ok=True)
    fig, ax, axt, fig_par = figure_create(
        rows=1, cols=1, plot_type=0, paper_col=paper_col, hgt_mltp=hgt_mltp, font=TGAExp.plot_font
    )
    for i, exp in enumerate(exps):
        ax[0].plot(
            exp.T,
            exp.mp_db,
            color=clrs[i],
            linestyle=lnstls[i],
            label=exp.label if exp.label else exp.name,
        )
        ax[0].fill_between(
            exp.T, exp.mp_db - exp.mp_db_std, exp.mp_db + exp.mp_db_std, color=clrs[i], alpha=0.3
        )
    figure_save(
        filename + "_tg",
        out_path_TGs,
        fig,
        ax,
        axt,
        fig_par,
        legend=legend_loc,
        x_lab="T [" + TGAExp.T_symbol + "]",
        y_lab=TGAExp.TG_lab,
        grid=TGAExp.plot_grid,
        tight_layout=True,
        x_lim=x_lim,
        y_lim=y_lim,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        annotate_lttrs=annotate_lttrs,
        save_as_pdf=save_as_pdf,
        save_as_svg=save_as_svg,
    )


def dtg_multi_plotOLD(
    exps,
    filename="Fig",
    paper_col=0.78,
    hgt_mltp=1.25,
    x_lim=None,
    y_lim=None,
    x_ticks=None,
    y_ticks=None,
    plt_gc=False,
    gc_Tlim=300,
    annotate_lttrs=False,
    save_as_pdf=False,
    save_as_svg=False,
):
    """
    Plot multiple DTG curves for a list of experiments.

    Args:
        exps (list): List of TGAExp objects representing the experiments.
        filename (str, optional): Name of the output file. Defaults to 'Fig'.
        paper_col (float, optional): Color of the plot background. Defaults to 0.78.
        hgt_mltp (float, optional): Height multiplier for the plot. Defaults to 1.25.
        x_lim (tuple, optional): Limits of the x-axis. Defaults to None.
        y_lim (tuple, optional): Limits of the y-axis. Defaults to None.
        y_ticks (list, optional): Custom y-axis tick labels. Defaults to None.
        lttrs (bool, optional): Whether to annotate letters on the plot. Defaults to False.
        plt_gc (bool, optional): Whether to plot the GC-MS maximum temperature line. Defaults to False.
        gc_Tlim (int, optional): Maximum temperature for the GC-MS line. Defaults to 300.
        save_as_pdf (bool, optional): Whether to save the plot as a PDF file. Defaults to False.
        save_as_svg (bool, optional): Whether to save the plot as an SVG file. Defaults to False.

    Returns:
        None
    """

    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    out_path = plib.Path(exps[0].out_path, "MultiSamplePlots")
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax, axt, fig_par = figure_create(
        rows=1, cols=1, plot_type=0, paper_col=paper_col, hgt_mltp=hgt_mltp, font=TGAExp.plot_font
    )
    for i, exp in enumerate(exps):
        ax[0].plot(
            exp.T_dtg,
            exp.dtg_db,
            color=clrs[i],
            linestyle=lnstls[i],
            label=exp.label if exp.label else exp.name,
        )
        ax[0].fill_between(
            exp.T_dtg,
            exp.dtg_db - exp.dtg_db_std,
            exp.dtg_db + exp.dtg_db_std,
            color=clrs[i],
            alpha=0.3,
        )
    if plt_gc:
        ax[0].vlines(
            gc_Tlim,
            ymin=y_lim[0],
            ymax=y_lim[1],
            linestyle=lnstls[1],
            color=clrs[7],
            label="T$_{max GC-MS}$",
        )
    ax[0].legend(loc="lower right")
    figure_save(
        filename + "_dtg",
        out_path,
        fig,
        ax,
        axt,
        fig_par,
        y_lab=TGAExp.DTG_lab,
        x_lab="T [" + TGAExp.T_symbol + "]",
        grid=TGAExp.plot_grid,
        tight_layout=True,
        x_lim=x_lim,
        y_lim=y_lim,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        annotate_lttrs=annotate_lttrs,
        save_as_pdf=save_as_pdf,
        save_as_svg=save_as_svg,
    )


def proximate_multi_plotOLD(
    exps,
    filename="Prox",
    smpl_labs=None,
    xlab_rot=0,
    paper_col=0.8,
    hgt_mltp=1.5,
    bboxtoanchor=True,
    x_anchor=1.13,
    y_anchor=1.02,
    legend_loc="best",
    y_lim=[0, 100],
    yt_lim=[0, 1],
    y_ticks=None,
    yt_ticks=None,
    y_lab="mass fraction [wt%]",
    yt_lab="mean TG deviation [%]",
    annotate_lttrs=False,
    save_as_pdf=False,
    save_as_svg=False,
):
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
    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    out_path = plib.Path(exps[0].out_path, "MultiSamplePlots")
    out_path.mkdir(parents=True, exist_ok=True)
    vars_bar = ["Moisture (stb)", "VM (db)", "FC (db)", "Ash (db)"]
    vars_scat = ["Mean TG dev."]
    if smpl_labs:
        labels = smpl_labs
    else:
        labels = [exp.label for exp in exps]
    df_ave = pd.DataFrame(columns=vars_bar, index=labels)
    df_std = pd.DataFrame(columns=vars_bar, index=labels)
    df_ave["Moisture (stb)"] = [exp.moist_ar for exp in exps]
    df_ave["VM (db)"] = [exp.vm_db for exp in exps]
    df_ave["FC (db)"] = [exp.fc_db for exp in exps]
    df_ave["Ash (db)"] = [exp.ash_db for exp in exps]
    df_std["Moisture (stb)"] = [exp.moist_ar_std for exp in exps]
    df_std["VM (db)"] = [exp.vm_db_std for exp in exps]
    df_std["FC (db)"] = [exp.fc_db_std for exp in exps]
    df_std["Ash (db)"] = [exp.ash_db_std for exp in exps]

    aveTG = [exp.AveTGstd_p for exp in exps]
    fig, ax, axt, fig_par = figure_create(
        1, 1, 1, paper_col=paper_col, hgt_mltp=hgt_mltp, font=TGAExp.plot_font
    )
    df_ave.plot(
        kind="bar",
        ax=ax[0],
        yerr=df_std,
        capsize=2,
        width=0.85,
        ecolor="k",
        edgecolor="black",
        rot=xlab_rot,
    )
    bars = ax[0].patches
    patterns = [None, "//", "...", "--"]  # set hatch patterns in correct order
    hatches = []  # list for hatches in the order of the bars
    for h in patterns:  # loop over patterns to create bar-ordered hatches
        for i in range(int(len(bars) / len(patterns))):
            hatches.append(h)
    # loop over bars and hatches to set hatches in correct order
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axt[0].errorbar(
        x=df_ave.index,
        y=aveTG,
        linestyle="None",
        marker=mrkrs[0],
        color=clrs[4],
        markersize=10,
        markeredgecolor="k",
        label="Mean TG dev.",
    )
    hnd_ax, lab_ax = ax[0].get_legend_handles_labels()
    hnd_axt, lab_axt = axt[0].get_legend_handles_labels()
    if bboxtoanchor:  # legend goes outside of plot area
        ax[0].legend(
            hnd_ax + hnd_axt,
            lab_ax + lab_axt,
            loc="upper left",
            bbox_to_anchor=(x_anchor, y_anchor),
        )
    else:  # legend is inside of plot area
        ax[0].legend(hnd_ax + hnd_axt, lab_ax + lab_axt, loc=legend_loc)
    if xlab_rot != 0:
        ax[0].set_xticklabels(df_ave.index, rotation=xlab_rot, ha="right", rotation_mode="anchor")
    figure_save(
        filename + "_prox",
        out_path,
        fig,
        ax,
        axt,
        fig_par,
        tight_layout=True,
        y_lab=y_lab,
        yt_lab=yt_lab,
        grid=TGAExp.plot_grid,
        legend=None,
        y_lim=y_lim,
        yt_lim=yt_lim,
        y_ticks=y_ticks,
        yt_ticks=yt_ticks,
        annotate_lttrs=annotate_lttrs,
        save_as_pdf=save_as_pdf,
        save_as_svg=save_as_svg,
    )


def oxidation_multi_plotOLD(
    exps,
    filename="Oxidations",
    smpl_labs=None,
    xlab_rot=0,
    paper_col=0.8,
    hgt_mltp=1.5,
    bboxtoanchor=True,
    x_anchor=1.13,
    y_anchor=1.02,
    legend_loc="best",
    y_lim=None,
    yt_lim=None,
    y_ticks=None,
    yt_ticks=None,
    annotate_lttrs=False,
    save_as_pdf=False,
    save_as_svg=False,
):
    """
    Generate a multi-plot for oxidation analysis.

    Parameters:
    - exps (list): List of experiments to be plotted.
    - filename (str): Name of the output file (default: "Oxidations").
    - smpl_labs (list): List of sample labels (default: None).
    - xlab_rot (int): Rotation angle of x-axis labels (default: 0).
    - paper_col (float): Color of the plot background (default: 0.8).
    - hgt_mltp (float): Height multiplier of the plot (default: 1.5).
    - bboxtoanchor (bool): Whether to place the legend outside the plot area (default: True).
    - x_anchor (float): X-coordinate of the legend anchor point (default: 1.13).
    - y_anchor (float): Y-coordinate of the legend anchor point (default: 1.02).
    - legend_loc (str): Location of the legend (default: 'best').
    - y_lim (tuple): Limits of the y-axis (default: None).
    - yt_lim (tuple): Limits of the twin y-axis (default: None).
    - y_ticks (list): Custom tick positions for the y-axis (default: None).
    - yt_ticks (list): Custom tick positions for the twin y-axis (default: None).

    Returns:
    - None
    """

    for exp in exps:
        if not exp.oxidation_computed:
            exp.oxidation_analysis()
    out_path = plib.Path(exps[0].out_path, "MultiSamplePlots")
    out_path.mkdir(parents=True, exist_ok=True)
    vars_bar = ["T$_i$", "T$_p$", "T$_b$"]
    vars_scat = ["S (combustibility index)"]
    if smpl_labs:
        labels = smpl_labs
    else:
        labels = [exp.label for exp in exps]
    df_ave = pd.DataFrame(columns=vars_bar, index=labels)
    df_std = pd.DataFrame(columns=vars_bar, index=labels)
    df_ave["T$_i$"] = [exp.Ti for exp in exps]
    df_ave["T$_p$"] = [exp.Tp for exp in exps]
    df_ave["T$_b$"] = [exp.Tb for exp in exps]
    df_std["T$_i$"] = [exp.Ti_std for exp in exps]
    df_std["T$_p$"] = [exp.Tp_std for exp in exps]
    df_std["T$_b$"] = [exp.Tb_std for exp in exps]

    S_combs = [exp.S for exp in exps]
    S_combs_std = [exp.S_std for exp in exps]
    fig, ax, axt, fig_par = figure_create(
        1, 1, 1, paper_col=paper_col, hgt_mltp=hgt_mltp, font=TGAExp.plot_font
    )
    df_ave.plot(
        kind="bar",
        ax=ax[0],
        yerr=df_std,
        capsize=2,
        width=0.85,
        ecolor="k",
        edgecolor="black",
        rot=xlab_rot,
    )
    bars = ax[0].patches
    patterns = [None, "//", "..."]  # set hatch patterns in the correct order
    hatches = []  # list for hatches in the order of the bars
    for h in patterns:  # loop over patterns to create bar-ordered hatches
        for i in range(int(len(bars) / len(patterns))):
            hatches.append(h)
    # loop over bars and hatches to set hatches in correct order
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axt[0].errorbar(
        x=df_ave.index,
        y=S_combs,
        yerr=S_combs_std,
        linestyle="None",
        marker=mrkrs[0],
        ecolor="k",
        capsize=2,
        markeredgecolor="k",
        color=clrs[3],
        markersize=10,
        label="S",
    )
    hnd_ax, lab_ax = ax[0].get_legend_handles_labels()
    hnd_axt, lab_axt = axt[0].get_legend_handles_labels()
    if bboxtoanchor:  # legend goes outside of plot area
        ax[0].legend(
            hnd_ax + hnd_axt,
            lab_ax + lab_axt,
            loc="upper left",
            bbox_to_anchor=(x_anchor, y_anchor),
        )
    else:  # legend is inside of plot area
        ax[0].legend(hnd_ax + hnd_axt, lab_ax + lab_axt, loc=legend_loc)
    if xlab_rot != 0:
        ax[0].set_xticklabels(df_ave.index, rotation=xlab_rot, ha="right", rotation_mode="anchor")
    figure_save(
        filename + "_oxidation",
        out_path,
        fig,
        ax,
        axt,
        fig_par,
        tight_layout=True,
        legend=None,
        yt_lab="S (combustion index) [-]",
        y_lab="T [" + TGAExp.T_symbol + "]",
        grid=TGAExp.plot_grid,
        y_lim=y_lim,
        yt_lim=yt_lim,
        y_ticks=y_ticks,
        yt_ticks=yt_ticks,
        annotate_lttrs=annotate_lttrs,
        save_as_pdf=save_as_pdf,
        save_as_svg=save_as_svg,
    )


def soliddist_multi_plotOLD(
    exps,
    filename="Dist",
    hgt_mltp=1.25,
    paper_col=0.78,
    labels=None,
    x_lim=None,
    y_lim=[[0, 1000], [0, 100]],
    x_ticks=None,
    y_ticks=None,
    print_dfs=True,
    annotate_lttrs=False,
    save_as_pdf=False,
    save_as_svg=False,
):
    """
    Plot the solid distribution for multiple experiments.

    Args:
        exps (list): List of Experiment objects.
        filename (str, optional): Name of the output file. Defaults to "Dist".
        hgt_mltp (float, optional): Height multiplier for the plot. Defaults to 1.25.
        paper_col (float, optional): Color of the plot background. Defaults to .78.
        labels (list, optional): List of labels for the experiments. Defaults to None.
        lttrs (bool, optional): Whether to annotate letters on the plot. Defaults to False.
        x_lim (list, optional): Limits for the x-axis. Defaults to None.
        y_lim (list, optional): Limits for the y-axis. Defaults to [[0, 1000], [0, 100]].
        y_ticks (list, optional): Custom tick locations for the y-axis. Defaults to None.
        print_dfs (bool, optional): Whether to print the dataframes. Defaults to True.
    """

    for exp in exps:
        if not exp.soliddist_computed:
            exp.soliddist_analysis()
    out_path = plib.Path(exps[0].out_path, "MultiSamplePlots")
    out_path.mkdir(parents=True, exist_ok=True)
    if not labels:  # try with labels and use name if no label is given
        labels = [exp.label if exp.label else exp.name for exp in exps]
    fig, ax, axt, fig_par = figure_create(
        rows=2, cols=1, plot_type=0, paper_col=paper_col, hgt_mltp=hgt_mltp, font=TGAExp.plot_font
    )
    for i, exp in enumerate(exps):
        ax[0].plot(exp.time, exp.T, color=clrs[i], linestyle=lnstls[i], label=labels[i])
        ax[0].fill_between(exp.time, exp.T - exp.T_std, exp.T + exp.T_std, color=clrs[i], alpha=0.3)
        ax[1].plot(exp.time, exp.mp_db, color=clrs[i], linestyle=lnstls[i], label=labels[i])
        ax[1].fill_between(
            exp.time, exp.mp_db - exp.mp_db_std, exp.mp_db + exp.mp_db_std, color=clrs[i], alpha=0.3
        )
        for tm, mp, dmp in zip(exp.time_dist, exp.loc_dist, exp.dmp_dist):
            ax[1].annotate(
                str(np.round(dmp, 0)) + "%",
                ha="center",
                va="top",
                xy=(tm - 10, mp + 1),
                fontsize=9,
                color=clrs[i],
            )
        ax[0].legend(loc="upper left")
        # ax[1].legend(loc='center left')
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
        x_lim=x_lim,
        y_lim=y_lim,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        annotate_lttrs=annotate_lttrs,
        save_as_pdf=save_as_pdf,
        save_as_svg=save_as_svg,
    )


def cscd_multi_plotOLD(
    exps,
    filename="Fig",
    paper_col=0.78,
    hgt_mltp=1.25,
    x_lim=None,
    clrs_cscd=False,
    y_lim0_cscd=-8,
    shifts_cscd=np.asarray([0, 11, 5, 10, 10, 10, 11]),
    peaks_cscd=None,
    peak_names=None,
    dh_names_cscd=0.1,
    loc_names_cscd=130,
    hgt_mltp_cscd=1.5,
    legend_cscd="lower right",
    y_values_cscd=[-10, 0],
    annotate_lttrs=False,
    save_as_pdf=False,
    save_as_svg=False,
):
    """
    Generate a cascaded multi-plot for a list of experiments.

    Args:
        exps (list): List of experiments.
        filename (str, optional): Filename for the saved plot. Defaults to 'Fig'.
        paper_col (float, optional): Width of the plot in inches. Defaults to 0.78.
        hgt_mltp (float, optional): Height multiplier for the plot. Defaults to 1.25.
        x_lim (tuple, optional): X-axis limits for the plot. Defaults to None.
        clrs_cscd (bool, optional): Flag to use custom colors for each experiment. Defaults to False.
        y_lim0_cscd (int, optional): Starting value for the y-axis limits. Defaults to -8.
        shifts_cscd (ndarray, optional): Array of shift values for each experiment. Defaults to np.asarray([0, 11, 5, 10, 10, 10, 11]).
        peaks_cscd (list, optional): List of peaks for each experiment. Defaults to None.
        peak_names (list, optional): List of names for each peak. Defaults to None.
        dh_names_cscd (float, optional): Shift value for peak names. Defaults to 0.1.
        loc_names_cscd (int, optional): Location for annotating experiment names. Defaults to 130.
        hgt_mltp_cscd (float, optional): Height multiplier for the cascaded plot. Defaults to 1.5.
        legend_cscd (str, optional): Location of the legend. Defaults to 'lower right'.
        y_values_cscd (list, optional): List of y-axis tick values. Defaults to [-10, 0].
        lttrs (bool, optional): Flag to annotate letters for each experiment. Defaults to False.
        save_as_pdf (bool, optional): Flag to save the plot as a PDF file. Defaults to False.
        save_as_svg (bool, optional): Flag to save the plot as an SVG file. Defaults to False.

    Returns:
        None
    """
    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    out_path = plib.Path(exps[0].out_path, "MultiSamplePlots")
    out_path.mkdir(parents=True, exist_ok=True)
    labels = [exp.label if exp.label else exp.name for exp in exps]
    y_lim_cscd = [y_lim0_cscd, np.sum(shifts_cscd)]
    dh = np.cumsum(shifts_cscd)
    fig, ax, axt, fig_par = figure_create(
        1, 1, paper_col=0.78, hgt_mltp=hgt_mltp_cscd, font=TGAExp.plot_font
    )
    for n, exp in enumerate(exps):
        if clrs_cscd:
            ax[0].plot(exp.T_dtg, exp.dtg_db + dh[n], color=clrs[n], linestyle=lnstls[0])
            ax[0].fill_between(
                exp.T_fit,
                exp.dtg_db - exp.dtg_db_std + dh[n],
                exp.dtg_db + exp.dtg_db_std + dh[n],
                color=clrs[n],
                alpha=0.3,
            )
        else:
            ax[0].plot(exp.T_dtg, exp.dtg_db + dh[n], color="k", linestyle=lnstls[0])
            ax[0].fill_between(
                exp.T_dtg,
                exp.dtg_db - exp.dtg_db_std + dh[n],
                exp.dtg_db + exp.dtg_db_std + dh[n],
                color="k",
                alpha=0.3,
            )
        ax[0].annotate(
            labels[n],
            ha="left",
            va="bottom",
            xy=(
                loc_names_cscd,
                exp.dtg_db[np.argmax(exp.T_dtg > loc_names_cscd)] + dh[n] + dh_names_cscd,
            ),
        )
    if peaks_cscd:
        for p, peak in enumerate(peaks_cscd):
            if peak:  # to allow to use same markers by skipping peaks
                ax[0].plot(
                    peak[0],
                    peak[1],
                    linestyle="None",
                    marker=mrkrs[p],
                    color="k",
                    label=peak_names[p],
                )
    if y_values_cscd:
        ax[0].set_yticks(y_values_cscd)
    else:
        ax[0].set_yticks([])
    figure_save(
        filename + "_cscd",
        out_path,
        fig,
        ax,
        axt,
        fig_par,
        legend=legend_cscd,
        x_lab="T [" + TGAExp.T_symbol + "]",
        y_lab=TGAExp.DTG_lab,
        x_lim=x_lim,
        y_lim=y_lim_cscd,
        grid=TGAExp.plot_grid,
        annotate_lttrs=annotate_lttrs,
        save_as_pdf=save_as_pdf,
        save_as_svg=save_as_svg,
    )


def KAS_analysis(exps, ramps, alpha=np.arange(0.05, 0.9, 0.05)):
    """
    Perform KAS (Kissinger-Akahira-Sunose) analysis on a set of experiments.

    Args:
        exps (list): List of Experiment objects representing the experiments to analyze.
        ramps (list): List of ramp values used for each experiment.
        alpha (numpy.ndarray, optional): Array of alpha values to investigate. Defaults to np.arange(0.05, .9, 0.05).

    Returns:
        dict: Dictionary containing the results of the KAS analysis, including the activation energy (Ea),
              the standard deviation of the activation energy (Ea_std), the alpha values, the ramp values,
              the x matrix, the y matrix, the fitted functions, and the name of the analysis.

    Raises:
        None
    """

    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    R_gas = 8.314462618
    n_ramps = len(ramps)  # number of ramp used for feedstock
    n_alpha = len(alpha)  # number of alpha investigated
    v_a = np.zeros(n_alpha, dtype=int)
    Ea = np.zeros(n_alpha)
    Ea_std = np.zeros(n_alpha)
    v_fit = []
    v_res_fit = np.zeros(n_alpha)
    r_sqrd_fit = np.zeros(n_alpha)
    ymatr = np.zeros((n_alpha, n_ramps))
    xmatr = np.zeros((n_alpha, n_ramps))
    for e, exp in enumerate(exps):
        # build the two matrixes
        if exp.T_unit == "Kelvin":
            TK = exp.T_dtg
        elif exp.T_unit == "Celsius":
            TK = exp.T_dtg + 273.15
        mp_db_dtg = exp.mp_db_dtg
        mdaf = (mp_db_dtg - np.min(mp_db_dtg)) / (np.max(mp_db_dtg) - np.min(mp_db_dtg))
        a = 1 - mdaf
        for i in range(n_alpha):
            v_a[i] = np.argmax(a > alpha[i])
        xmatr[:, e] = 1 / TK[v_a] * 1000
        for i in range(n_alpha):
            # BETA IS IN MINUTE HERE
            ymatr[i, e] = np.log(ramps[e] / TK[v_a[i]] ** 2)

    for i in range(n_alpha):
        p, cov = np.polyfit(xmatr[i, :], ymatr[i, :], 1, cov=True)
        v_fit.append(np.poly1d(p))
        v_res_fit = ymatr[i, :] - v_fit[i](xmatr[i, :])
        r_sqrd_fit[i] = 1 - (
            np.sum(v_res_fit**2) / np.sum((ymatr[i, :] - np.mean(ymatr[i, :])) ** 2)
        )
        Ea[i] = -v_fit[i][1] * R_gas
        Ea_std[i] = np.sqrt(cov[0][0]) * R_gas
    # the name is obtained supposing its Sample-Ox# with # = rate
    if "-Ox" in exps[0].name:
        name = exps[0].name.split("-Ox")[0]
    else:
        name = exps[0].name.split("Ox")[0]
    if name == "":
        name = "SampleNameN.A."
    kas = {
        "Ea": Ea,
        "Ea_std": Ea_std,
        "alpha": alpha,
        "ramps": ramps,
        "xmatr": xmatr,
        "ymatr": ymatr,
        "v_fit": v_fit,
        "name": name,
    }
    for e, exp in enumerate(exps):
        exp.kas = kas
        exp.KAS_computed = True
    return kas


def KAS_plot_isolinesOLD(
    exps,
    kas_names=None,
    filename="KAsIso",
    paper_col=0.78,
    hgt_mltp=1.25,
    x_lim=None,
    y_lim=None,
    annt_names=True,
    leg_cols=1,
    bboxtoanchor=True,
    x_anchor=1.13,
    y_anchor=1.02,
    legend_loc="best",
    annotate_lttrs=False,
    save_as_pdf=False,
    save_as_svg=False,
):
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

    for exp in exps:
        if not exp.KAS_computed:
            print("need KAS analysis")
    out_path = plib.Path(exps[0].out_path, "KAS")
    out_path.mkdir(parents=True, exist_ok=True)
    kass = [exp.kas for exp in exps]
    xmatrs = [kas["xmatr"] for kas in kass]
    # for plots
    if kas_names is None:
        kas_names = [kas["name"] for kas in kass]

    n_exp = len(kass)  # number of feedstocks
    alphas = [kas["alpha"] for kas in kass]
    if not all(np.array_equal(alphas[0], a) for a in alphas):
        print("samples have been analyzed at different alphas")

    else:
        alpha = alphas[0]
    n_alpha = len(alpha)

    x = np.linspace(
        np.min([np.min(xmatr) for xmatr in xmatrs]),
        np.max([np.max(xmatr) for xmatr in xmatrs]),
        100,
    )
    rows = n_exp
    cols = 1
    ax_for_legend = 0
    if n_exp > 3:
        rows = round(rows / 2 + 0.01)
        cols += 1
        paper_col *= 1.5
        ax_for_legend += 1
    fig, ax, axt, fig_par = figure_create(
        rows=rows,
        cols=cols,
        plot_type=0,
        paper_col=paper_col,
        hgt_mltp=hgt_mltp,
        font=TGAExp.plot_font,
    )
    for k, kas in enumerate(kass):
        ymaxiso = np.max(kas["ymatr"])
        yminiso = np.min(kas["ymatr"])
        for i in range(n_alpha):
            lab = r"$\alpha$=" + str(np.round(alpha[i], 2))
            xmin = np.argwhere(kas["v_fit"][i](x) < ymaxiso)[0][0]
            try:
                xmax = np.argwhere(kas["v_fit"][i](x) < yminiso)[0][0]
            except IndexError:
                xmax = 0
            newx = x[xmin:xmax]
            ax[k].plot(newx, kas["v_fit"][i](newx), color=clrs[i], linestyle=lnstls[i])
            ax[k].plot(
                kas["xmatr"][i, :],
                kas["ymatr"][i, :],
                color=clrs[i],
                linestyle="None",
                marker=mrkrs[i],
            )
            ax[k].plot([], [], color=clrs[i], linestyle=lnstls[i], marker=mrkrs[i], label=lab)
            hnd_ax, lab_ax = ax[k].get_legend_handles_labels()
        if annt_names:
            ax[k].annotate(
                kas_names[k],
                xycoords="axes fraction",
                xy=(0, 0),
                rotation=0,
                size="small",
                xytext=(0.02, 1.02),
            )
    if bboxtoanchor:  # legend goes outside of plot area

        ax[ax_for_legend].legend(
            ncol=leg_cols, loc="upper left", bbox_to_anchor=(x_anchor, y_anchor)
        )
    else:  # legend is inside of plot area
        ax[0].legend(ncol=leg_cols, loc=legend_loc)
    figure_save(
        filename + "_iso",
        out_path,
        fig,
        ax,
        axt,
        fig_par,
        x_lim=x_lim,
        y_lim=y_lim,
        x_lab="1000/T [1/K]",
        legend=None,
        annotate_lttrs=annotate_lttrs,
        y_lab=r"ln($\beta$/T$^{2}$)",
        tight_layout=False,
        grid=TGAExp.plot_grid,
    )


def KAS_plot_EaOLD(
    exps,
    kas_names=None,
    filename="KASEa",
    paper_col=0.78,
    hgt_mltp=1.25,
    x_lim=[0.1, 0.8],
    y_lim=[0, 300],
    y_ticks=None,
    leg_cols=1,
    bboxtoanchor=True,
    x_anchor=1.13,
    y_anchor=1.02,
    plot_type="scatter",
    legend="best",
    annotate_lttrs=False,
    save_as_pdf=False,
    save_as_svg=False,
):
    """
    Plot the activation energy (Ea) for multiple experiments.

    Parameters:
    - exps (list): List of experiments.
    - kas_names (list, optional): List of names for the KAS analysis. If not provided, the names will be extracted from the experiments.
    - filename (str, optional): Name of the output file. Default is 'KASEa'.
    - paper_col (float, optional): Color of the plot background. Default is 0.78.
    - hgt_mltp (float, optional): Height multiplier for the plot. Default is 1.25.
    - x_lim (list, optional): Limits of the x-axis. Default is [0.1, 0.8].
    - y_lim (list, optional): Limits of the y-axis. Default is [0, 300].
    - y_ticks (list, optional): Custom y-axis tick locations. Default is None.
    - annt_names (bool, optional): Whether to annotate the names of the experiments. Default is True.
    - annotate_lttrs (bool, optional): Whether to annotate the letters of the experiments. Default is False.
    - leg_cols (int, optional): Number of columns in the legend. Default is 1.
    - bboxtoanchor (bool, optional): Whether to place the legend outside of the plot area. Default is True.
    - x_anchor (float, optional): X-coordinate of the legend anchor. Default is 1.13.
    - y_anchor (float, optional): Y-coordinate of the legend anchor. Default is 2.02.
    - plot_type (str, optional): Type of plot. Can be 'scatter' or 'line'. Default is 'scatter'.
    - legend_loc (str, optional): Location of the legend. Default is 'best'.
    """

    for exp in exps:
        if not exp.KAS_computed:
            exp.KAS_analysis()
    out_path = plib.Path(exps[0].out_path, "KAS")
    out_path.mkdir(parents=True, exist_ok=True)
    kass = [exp.kas for exp in exps]
    if kas_names is None:
        kas_names = [kas["name"] for kas in kass]
    alphas = [kas["alpha"] for kas in kass]
    if not all(np.array_equal(alphas[0], a) for a in alphas):
        print("samples have been analyzed at different alphas")

    else:
        alpha = alphas[0]
    print(kas_names)
    # plot activation energy
    fig, ax, axt, fig_par = figure_create(
        rows=1, cols=1, plot_type=0, paper_col=paper_col, hgt_mltp=hgt_mltp, font=TGAExp.plot_font
    )
    for k, kas in enumerate(kass):
        if plot_type == "scatter":
            ax[0].errorbar(
                alpha,
                kas["Ea"],
                kas["Ea_std"],
                color="k",
                linestyle="None",
                capsize=3,
                ecolor=clrs[k],
            )
            ax[0].plot(
                alpha,
                kas["Ea"],
                color=clrs[k],
                linestyle="None",
                marker=mrkrs[k],
                label=kas_names[k],
            )
        elif plot_type == "line":
            ax[0].plot(alpha, kas["Ea"], color=clrs[k], linestyle=lnstls[k], label=kas_names[k])
            ax[0].fill_between(
                alpha,
                kas["Ea"] - kas["Ea_std"],
                kas["Ea"] + kas["Ea_std"],
                color=clrs[k],
                alpha=0.3,
            )
    if legend is not None:
        if bboxtoanchor:  # legend goes outside of plot area
            ax[0].legend(ncol=leg_cols, loc="upper left", bbox_to_anchor=(x_anchor, y_anchor))
        else:  # legend is inside of plot area
            ax[0].legend(ncol=leg_cols, loc=legend)
    figure_save(
        filename + "_Ea",
        out_path,
        fig,
        ax,
        axt,
        fig_par,
        x_lim=x_lim,
        y_lim=y_lim,
        annotate_lttrs=annotate_lttrs,
        y_ticks=y_ticks,
        x_lab=r"$\alpha$ [-]",
        tight_layout=True,
        y_lab=r"$E_{a}$ [kJ/mol]",
        grid=TGAExp.plot_grid,
    )
