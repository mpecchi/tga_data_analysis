import string
import pathlib as plib
import numpy as np
import seaborn as sns
from typing import Literal
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

lttrs: list[str] = list(string.ascii_lowercase)

# list with colors
clrs: list[tuple] = sns.color_palette("deep", 30)

# list with linestyles for plotting
lnstls: list[tuple] = [
    (0, ()),  # solid
    (0, (1, 1)),  # 'densely dotted'
    (0, (5, 1)),  # 'densely dashed'
    (0, (3, 1, 1, 1)),  # 'densely dashdotted'
    (0, (3, 1, 1, 1, 1, 1)),  # 'densely dashdotdotted'
    (0, (5, 5)),  # 'dashed'
    (0, (3, 5, 1, 5)),  # 'dashdotted'
    (0, (1, 5)),  # dotted
    (0, (3, 5, 1, 5, 1, 5)),  # 'dashdotdotted'
    (0, (1, 10)),  # 'loosely dotted'
    (0, (5, 10)),  # 'loosely dashed'
    (0, (3, 10, 1, 10)),  # 'loosely dashdotted'
    (0, (3, 10, 1, 10, 1, 10)),
    (0, ()),  # solid
    (0, (1, 1)),  # 'densely dotted'
    (0, (5, 1)),  # 'densely dashed'
    (0, (3, 1, 1, 1)),  # 'densely dashdotted'
    (0, (3, 1, 1, 1, 1, 1)),  # 'densely dashdotdotted'
    (0, (5, 5)),  # 'dashed'
    (0, (3, 5, 1, 5)),  # 'dashdotted'
    (0, (1, 5)),  # dotted
    (0, (3, 5, 1, 5, 1, 5)),  # 'dashdotdotted'
    (0, (1, 10)),  # 'loosely dotted'
    (0, (5, 10)),  # 'loosely dashed'
    (0, (3, 10, 1, 10)),  # 'loosely dashdotted'
    (0, (3, 10, 1, 10, 1, 10)),
]  # 'loosely dashdotdotted'

# list with markers for plotting
mrkrs: list[str] = [
    "o",
    "v",
    "X",
    "s",
    "p",
    "^",
    "P",
    "<",
    ">",
    "*",
    "d",
    "1",
    "2",
    "3",
    "o",
    "v",
    "X",
    "s",
    "p",
    "^",
    "P",
    "<",
    ">",
    "*",
    "d",
    "1",
    "2",
    "3",
]


def figure_create(
    rows: int = 1,
    cols: int = 1,
    plot_type: int = 0,
    paper_col: float = 1,
    hgt_mltp: float = 1,
    font: Literal["Dejavu Sans", "Times New Roman"] = "Dejavu Sans",
    sns_style: str = "ticks",
) -> tuple[Figure, list[Axes], None | list[Axes], list[float]]:
    """
    This function creates all the necessary objects to produce plots with
    replicable characteristics.

    Parameters
    ----------
    rows : int, optional
        Number of plot rows in the grid. The default is 1.
    cols : int, optional
        Number of plot columns in the grid. The default is 1.
    plot_type : int, optional
        One of the different plot types available. The default is 0.
        Plot types and their labels:
        0. Std: standard plot (single or grid rows x cols)
        1. Twin-x: secondary axis plot (single or grid rows x cols)
        5. Subplots with different heights
        6. Multiplot without internal x and y tick labels
        7. Multiplot without internal x tick labels
        8. Plot with specific distances between subplots and different heights
    paper_col : int, optional
        Single or double column size for the plot, meaning the actual space
        it will fit in a paper. The default is 1.
    hgt_mltp: float, optional
        Multiplies the figure height. Default is 1. Best using values between
        0.65 and 2. May not work with multiplot and paper_col=1 or out of the
        specified range.
    font: str, optional
        If the string 'Times' is given, it sets Times New Roman as the default
        font for the plot, otherwise the default Dejavu Sans is maintained.
        Default is 'Dejavu Sans'.
    sns_style: str, optional
        The style of the seaborn plot. The default is 'ticks'.

    Returns
    -------
    fig : object
        The figure object to be passed to fig_save.
    lst_ax : list of axis
        List of axis (it is a list even with 1 axis) on which to plot.
    lst_axt : list of axis
        List of secondary axis (it is a list even with 1 axis).
    fig_par : list of float
        List of parameters to reserve space around the plot canvas.

    Raises
    ------
    ValueError
        If cols > 2, which is not supported.

    """

    sns.set_palette("deep")
    if (
        font == "Times" or font == "Times New Roman"
    ):  # set Times New Roman as the plot font fot text
        # this may require the installation of the font package
        sns.set_style(sns_style, {"font.family": "Times New Roman"})
    else:  # leave Dejavu Sans (default) as the plot font fot text
        sns.set_style(sns_style)
    # single or double column in paperthat the figure will occupy
    if cols > 2:  # numer of columns (thus of plots in the figure)
        raise ValueError("\n fig_create: cols>2 not supported")

    # width of the figure in inches, it's fixed to keep the same text size
    # is 6, 9, 12 for 1, 1.5, and 3 paper_col (columns in paper)
    fig_wdt: float = 6 * paper_col  # width of the plot in inches
    fig_hgt: float = 4 * paper_col * rows / cols * hgt_mltp  # heigth of the figure in inches
    px: float = 0.06 * (6 / fig_wdt) * cols  # set px so that (A) fits the square
    py: float = px * fig_wdt / fig_hgt / cols * rows / hgt_mltp  # set py so that (A) fits
    # if more rows are added, it increases, but if cols areadded it decreases
    # to maintain the plot ratio
    # set plot margins
    sp_lab_wdt: float = 0.156 / paper_col  # hor. space for labels
    sp_nar_wdt: float = 0.02294 / paper_col  # space narrow no labels (horiz)
    sp_lab_hgt: float = 0.147 / paper_col / rows * cols / hgt_mltp  # space for labels (vert)
    sp_nar_hgt: float = 0.02 / paper_col / rows * cols / hgt_mltp  # space narrow no labels
    # (vert)
    # =========================================================================
    # # 0. Std: standard plot (single or grid rows x cols)
    # =========================================================================
    fig: Figure
    ax: Axes
    if plot_type == 0:
        fig, ax = plt.subplots(rows, cols, figsize=(fig_wdt, fig_hgt))
        if rows * cols == 1:  # only 1 plot
            lst_ax = [ax]  # create ax list for uniform iterations over 1 obj.
        elif rows * cols > 1:  # more than one plot
            lst_ax = [axs for axs in ax.flatten()]  # create list of axis
        lst_axt = None  # no secondary axis in this plot_type
        # horizontal space between plot in percentage
        sp_btp_wdt: float = 0.26 * paper_col**2 - 1.09 * paper_col + 1.35
        # vertical space between plot in percentage !!! needs DEBUG
        sp_btp_hgt: float = 0.2 / paper_col * cols / hgt_mltp
        # left, bottom, right, top, widthspace, heightspace
        fig_par: list[float] = [
            sp_lab_wdt,
            sp_lab_hgt,
            1 - sp_nar_wdt,
            1 - sp_nar_hgt,
            sp_btp_wdt,
            sp_btp_hgt,
            px,
            py,
        ]
    # =========================================================================
    # # 1. Twin-x: secondary axis plot (single or grid rows x cols)
    # =========================================================================
    elif plot_type == 1:
        fig, ax = plt.subplots(rows, cols, figsize=(fig_wdt, fig_hgt))
        if rows * cols == 1:  # only 1 plot
            lst_ax = [ax]  # create ax list for uniform iterations over 1 obj.
            lst_axt = [ax.twinx()]  # create a list with secondary axis object
        elif rows * cols > 1:  # more than one plot
            lst_ax = [axs for axs in ax.flatten()]  # create list of axis
            # create list of secondary twin axis
            lst_axt = [axs.twinx() for axs in ax.flatten()]
        # horizontal space between plot in percentage !!! needs DEBUG
        sp_btp_wdt: float = 1.36 * paper_col**2 - 5.28 * paper_col + 5.57
        # vertical space between plot in percentage !!! needs DEBUG
        sp_btp_hgt: float = 0.2 / paper_col * cols / hgt_mltp
        # left, bottom, right(DIFFERENT FROM STD), top, widthspace, heightspace
        fig_par: float = [
            sp_lab_wdt,
            sp_lab_hgt,
            1 - sp_lab_wdt,
            1 - sp_nar_hgt,
            sp_btp_wdt,
            sp_btp_hgt,
            px,
            py,
        ]

    return fig, lst_ax, lst_axt, fig_par


def figure_save(
    filename: str,
    out_path: plib.Path,
    fig: Figure,
    lst_ax: list[Axes],
    lst_axt: list[Axes] | None,
    fig_par: list[float],
    x_lab: str | None = None,
    y_lab: str | None = None,
    yt_lab: str | None = None,
    x_lim: float | None = None,
    y_lim: float | None = None,
    yt_lim: float | None = None,
    x_ticks: list[float] | None = None,
    y_ticks: list[float] | None = None,
    yt_ticks: list[float] | None = None,
    x_ticklabels: list[str] | None = None,
    y_ticklabels: list[str] | None = None,
    yt_ticklabels: list[str] | None = None,
    legend: None | str = None,
    ncol_leg: int = 1,
    annotate_lttrs: bool = False,
    annotate_lttrs_loc: Literal["up", "down"] = "down",
    save_as_pdf: bool = False,
    save_as_svg: bool = False,
    save_as_eps: bool = False,
    png_transparency: bool = False,
    tight_layout: bool = False,
    grid: bool = False,
    title: str | bool = False,
    set_size_inches: None | float | tuple[float, float] = None,
):
    """
    This function takes the obects created in fig_create and allows to modify
    their appeareance and saving the results.

    Parameters
    ----------
    filename : str
        name of figure. It is the name of the png od pfd file to be saved
    out_path : pathlib.Path object. path to the output folder.
    fig : figure object. created in fig_save.
    lst_ax : list of axis. Created in fig_create
    lst_axt : list of twin (secondary) axis. Created in fig_create
    fig_par : list of figure parameters for space settings
        left, bottom, right, top, widthspace, heightspace, px, py.
        Created in fig_create
    tight_layout : bool
        If True, ignore fig_par[0:6] and fit the figure to the tightest layout
        possible. Avoids to lose part of figure, but loses control of margins
    xLab : str.list, optional
        label of the x axis. The default is None.
        can be given as
        0. xLab=None: no axis gets an xlabel
        1. xLab='label': only one str, all axis get the same xlabel
        2. xLab=['label1', None, Label2, ...]: the list must have the size of
            lst_ax and contain labels and or None values. Each axis is
            assigned its label, where None is given, no label is set.
    yLab : str, optional
        label of the y axis. The default is None. Same options as xLab
    ytLab : str, optional
        label of the secondary y-axis. The default is None.
        Same options as xLab
    xLim : list of two values, list of lists, optional
        limits of x axis. The default is None.
        can be given as
        0. xLim=None: no axis gets a xlim
        1. xLab=[a,b]: all axis get the same xlim
        2. xLab=[[a,b], None, [c,d], ...]: the list must have the size of
            lst_ax and contain [a,b] and or None values. Each axis is
            assigned its limit, where None is given, no llimit is set.
    yLim : list of two values, optional
        limits of y axis. The default is None. Same options as xLim
    ytLim : list of two values, optional
        limits of secondary y axis. The default is None.
        Same options as xLim
    xTicks : list of int or float, optional
        list of tiks value to be shown on the axis. The default is None.
    yTicks : list of int or float, optional
        list of tiks value to be shown on the axis. The default is None.
    ytTicks : TYPE, optional
        list of tiks value to be shown on the axis. The default is None.
    legend : str, optional
        contains info on the legend location. To avoid printing the legend
        (also in case it is empty) set it to None.
        The default is 'best'.
    ncol_leg : int, optional
        number of columns in the legend. The default is 1.
    annotate_lttrs : bool, optional
        if True, each plot is assigned a letter between () in the lower left
        corner. The default is False. If a string is given, the string is used
        as the letter in the plot even for single plots.
    annotate_lttrs_loc: str.
        default is 'down', if 'up' is given, the letters are placed on the left
        top corner.
    pdf : bool, optional
        if True, the figure is saved also in pdf in the output folder.
        The default is False, so only a png file with 300dpi is saved
    transparency : bool, optional
        if True, background of PNG figure is transparent, defautls is False.
    """

    fig_adj_par = fig_par[0:6]
    if not any(fig_par[0:6]):  # True if all element in fig_par[0:6] are False
        tight_layout = True
    px = fig_par[6]
    py = fig_par[7]
    n_ax = len(lst_ax)  # number of ax objects
    # for xLab, yLab, ytLab creates a list with same length as n_ax.
    # only one value is given all axis are given the same label
    # if a list is given, each axis is given a different value, where False
    # is specified, no value is given to that particular axis
    vrbls = [x_lab, y_lab, yt_lab, legend]  # collect variables for iteration
    lst_x_lab, lst_y_lab, lst_yt_lab, lst_legend = (
        [],
        [],
        [],
        [],
    )  # create lists for iteration
    lst_vrbls = [lst_x_lab, lst_y_lab, lst_yt_lab, lst_legend]  # collect lists
    for vrbl, lst_vrbl in zip(vrbls, lst_vrbls):
        if vrbl is None:  # label is not given for any axis
            lst_vrbl[:] = [None] * n_ax
        else:  # label is given
            if np.size(vrbl) == 1:  # only one value is given
                if isinstance(vrbl, str):  # create a list before replicating it
                    lst_vrbl[:] = [vrbl] * n_ax  # each axis gets same label
                elif isinstance(vrbl, list):  # replicate the list
                    lst_vrbl[:] = vrbl * n_ax  # each axis gets same label
            elif np.size(vrbl) == n_ax:  # each axis has been assigned its lab
                lst_vrbl[:] = vrbl  # copy the label inside the list
            else:
                print(vrbl)
                print("Labels/legend size does not match axes number")
    # for xLim, yLim, ytLim creates a list with same length as n_ax.
    # If one list like [a,b] is given, all axis have the same limits, if a list
    # of the same length of the axis is given, each axis has its lim. Where
    # None is given, no lim is set on that axis
    vrbls = [
        x_lim,
        y_lim,
        yt_lim,
        x_ticks,
        y_ticks,
        yt_ticks,
        x_ticklabels,
        y_ticklabels,
        yt_ticklabels,
    ]  # collect variables for iteration
    (
        lst_x_lim,
        lst_y_lim,
        lst_yt_lim,
        lst_x_ticks,
        lst_y_ticks,
        lst_yt_ticks,
        lst_x_ticklabels,
        lst_y_ticklabels,
        lst_yt_ticklabels,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )  # create lists for iteration
    lst_vrbls = [
        lst_x_lim,
        lst_y_lim,
        lst_yt_lim,
        lst_x_ticks,
        lst_y_ticks,
        lst_yt_ticks,
        lst_x_ticklabels,
        lst_y_ticklabels,
        lst_yt_ticklabels,
    ]  # collect lists
    for vrbl, lst_vrbl in zip(vrbls, lst_vrbls):
        if vrbl is None:  # limit is not given for any axis
            lst_vrbl[:] = [None] * n_ax
        else:
            # if only list and None are in vrbl, it is [[], None, [], ..]
            # each axis has been assigned its limits
            if any([isinstance(v, (int, float, np.int32, str)) for v in vrbl]):
                temporary = []  # necessary to allow append on [:]
                for i in range(n_ax):
                    temporary.append(vrbl)  # give it to all axis
                lst_vrbl[:] = temporary
            else:  # xLim=[[a,b], None, ...] = [list, bool] # no float
                lst_vrbl[:] = vrbl  # a lim for each axis is already given
    # loops over each axs in the ax array and set the different properties
    for i, axs in enumerate(lst_ax):
        # for each property, if the variable is not false, it is set
        if lst_x_lab[i] is not None:
            axs.set_xlabel(lst_x_lab[i])
        if lst_y_lab[i] is not None:
            axs.set_ylabel(lst_y_lab[i])
        if lst_x_lim[i] is not None:
            axs.set_xlim(
                [
                    lst_x_lim[i][0] * (1 + px) - px * lst_x_lim[i][1],
                    lst_x_lim[i][1] * (1 + px) - px * lst_x_lim[i][0],
                ]
            )
        if lst_y_lim[i] is not None:
            axs.set_ylim(
                [
                    lst_y_lim[i][0] * (1 + py) - py * lst_y_lim[i][1],
                    lst_y_lim[i][1] * (1 + py) - py * lst_y_lim[i][0],
                ]
            )
        if lst_x_ticks[i] is not None:
            axs.set_xticks(lst_x_ticks[i])
        if lst_y_ticks[i] is not None:
            axs.set_yticks(lst_y_ticks[i])
        if lst_x_ticklabels[i] is not None:
            axs.set_xticklabels(lst_x_ticklabels[i])
        if lst_y_ticklabels[i] is not None:
            axs.set_yticklabels(lst_y_ticklabels[i])
        if grid:
            axs.grid(True)
        if annotate_lttrs is not False:
            if annotate_lttrs_loc == "down":
                y_lttrs = py / px * 0.02
            elif annotate_lttrs_loc == "up":
                y_lttrs = 1 - py
            if n_ax == 1:  # if only one plot is given, do not put the letters
                axs.annotate(
                    "(" + annotate_lttrs + ")",
                    xycoords="axes fraction",
                    xy=(0, 0),
                    rotation=0,
                    size="large",
                    xytext=(0, y_lttrs),
                    weight="bold",
                )
            elif n_ax > 1:  # if only one plot is given, do not put the letters
                try:  # if specific letters are provided
                    axs.annotate(
                        "(" + annotate_lttrs[i] + ")",
                        xycoords="axes fraction",
                        xy=(0, 0),
                        rotation=0,
                        size="large",
                        xytext=(0, y_lttrs),
                        weight="bold",
                    )
                except TypeError:  # if no specific letters, use lttrs
                    axs.annotate(
                        "(" + lttrs[i] + ")",
                        xycoords="axes fraction",
                        xy=(0, 0),
                        rotation=0,
                        size="large",
                        xytext=(0, y_lttrs),
                        weight="bold",
                    )

    # if secondary (twin) axis are given, set thier properties
    if lst_axt is not None:
        for i, axst in enumerate(lst_axt):
            axst.grid(False)  # grid is always false on secondaty axis
            # for each property, if the variable is not false, it is set
            if lst_yt_lab[i] is not None:
                axst.set_ylabel(lst_yt_lab[i])
            if lst_yt_lim[i] is not None:
                axst.set_ylim(
                    [
                        lst_yt_lim[i][0] * (1 + py) - py * lst_yt_lim[i][1],
                        lst_yt_lim[i][1] * (1 + py) - py * lst_yt_lim[i][0],
                    ]
                )
            if lst_yt_ticks[i] is not None:
                axst.set_yticks(lst_yt_ticks[i])
            if lst_yt_ticklabels[i] is not None:
                axst.set_yticklabels(lst_yt_ticklabels[i])
    # create a legend merging the entries for each couple of ax and axt
    if any(lst_legend):
        if lst_axt is None:  # with no axt, only axs in ax needs a legend
            for i, axs in enumerate(lst_ax):
                axs.legend(loc=lst_legend[i], ncol=ncol_leg)
        else:  # merge the legend for each couple of ax and axt
            i = 0
            for axs, axst in zip(lst_ax, lst_axt):
                hnd_ax, lab_ax = axs.get_legend_handles_labels()
                hnd_axt, lab_axt = axst.get_legend_handles_labels()
                axs.legend(hnd_ax + hnd_axt, lab_ax + lab_axt, loc=lst_legend[i], ncol=ncol_leg)
                i += 1
    try:
        fig.align_labels()  # align labels of subplots, needed only for multi plot
    except AttributeError:
        print("align_labels not performed")
    # set figure margins and save the figure in the output folder
    if set_size_inches:
        fig.set_size_inches(set_size_inches)
    if tight_layout is False:  # if margins are given sets margins and save
        fig.subplots_adjust(*fig_adj_par[0:6])  # set margins
        plt.savefig(
            plib.Path(out_path, filename + ".png"),
            dpi=300,
            transparent=png_transparency,
        )
        if save_as_pdf is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + ".pdf"))
        if save_as_svg is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + ".svg"))
        if save_as_eps is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + ".eps"))
    else:  # margins are not given, use a tight layout option and save
        plt.savefig(
            plib.Path(out_path, filename + ".png"),
            bbox_inches="tight",
            dpi=300,
            transparent=png_transparency,
        )
        if save_as_pdf is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + ".pdf"), bbox_inches="tight")
        if save_as_svg is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + ".svg"), bbox_inches="tight")
        if save_as_eps is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + ".eps"), bbox_inches="tight")
    # add the title after saving, so it's only visible in the console
    if title is True:
        lst_ax[0].annotate(
            filename,
            xycoords="axes fraction",
            size="small",
            xy=(0, 0),
            xytext=(0.05, 0.95),
            clip_on=True,
        )
