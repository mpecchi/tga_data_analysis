# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:28:04 2022

@author: mp933
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib as plib
import pandas as pd
from scipy.signal import savgol_filter as SavFil
from lmfit.models import GaussianModel, LinearModel

# for plot cosmetics
font_scale = 1.25
font='Times'
font='Dejavu Sans'
sns.set("notebook", font_scale=1.25)
sns_style = "ticks"
# list with colors
palette = 'deep'
n_of_colors = 30
clrs = sns.color_palette(palette, n_of_colors)
# list with linestyles for plotting
lnstls = [(0, ()),  # solid
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
            (0, (3, 10, 1, 10, 1, 10)),]  # 'loosely dashdotdotted'
# list with letters for plotting
lttrs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
         'o', 'p', 'q']
# list with markers for plotting
mrkrs = ["o", "v", "X", "s", "p", "^", "P", "<", ">", "*", "d", "1", "2", "3",
         "o", "v", "X", "s", "p", "^", "P", "<", ">", "*", "d", "1", "2", "3"]

def PathsCreate(subfolder=''):
    ''' This function creates 2 folder paths (independently of the os). \
        For each path checks the folder existence, if False it creates it.
    in_path : pathlib object
        path to the Input folder.
    out_path : pathlib object
        path to the Output folder.'''
    try:  # start from script folder
        script_path = plib.Path(__file__).parents[0]  # script folder path
    except NameError:  # if a cell is run alone and the __file__ is not available
        script_path = plib.Path.cwd()  # the current working directory
    # create all necessary paths to subfolders starting from the script one
    if subfolder == '_test':  # only for the _test work in the _test folder
        in_path = plib.Path(script_path, "_test")
        out_path = plib.Path(script_path, "_test", 'Output')
    else:  # in all other cases
        in_path = plib.Path(script_path, "Input", subfolder)
        out_path = plib.Path(script_path, "Input", subfolder, 'Output')
    # check existence of each folders and create them if missing
    plib.Path(in_path).mkdir(parents=True, exist_ok=True)
    plib.Path(out_path).mkdir(parents=True, exist_ok=True)

    return in_path, out_path  # returns the two object-paths


# used to make plot all Times New Roman
# def FigCreate(rows=1, cols=1, plot_type=0, paper_col=1,
#               gridspec_hgt_fr=None,
#               gridspec_wdt_fr=None, hgt_mltp=1, font='DejaVu Sans',
#               sns_style='ticks'):
def FigCreate(rows=1, cols=1, plot_type=0, paper_col=1,
              gridspec_hgt_fr=None,
              gridspec_wdt_fr=None, hgt_mltp=1, font=font,
              sns_style='ticks'):
    """
    This function creates all the necessary objects to produce plots with
    replicable characteristics.

    Parameters
    ----------
    rows : int, optional
        number of plot rows in the grid. The default is 1.
    cols : int, optional
        number of plot colums in the grid. The default is 1.
    plot_type : int, optional
        one of the different plots available. The default is 0.
        Plot types and their labels:
        0. Std: standard plot (single or grid rows x cols)
        1. Twin-x: secondary axis plot (single or grid rows x cols)
        5. Subplots with different heights
        6. multiplot without internal x and y tickslabels
        7. multiplot without internal x tickslabels
        8. plot with specific distances between subplots and diffeerent heights
    paper_col : int, optional
        single or double column size for the plot, meaning the actual space
        it will fit in a paper. The default is 1.
    gridspec_wdt_ratio : list of float, optional
        for multiple cols, list of relative width of each subplot.
        The default is None.
    no_par : bool, optional
        if True, no size setting parameters are passed. The default is None.
    font: str
        if the str'Times' is given, it sets Times New Roman as the default
        font for the plot, otherwise the default Dejavu Sans is maintained.
        Default is 'Dejavu Sans'
    hgt_mltp: float
        multiplies the fig height. default is to 1. best using values between
        .65 and 2. may not work with multiplot and paper_col=1 or out of the
        specified range
    Returns
    -------
    fig : object
        the figure object to be passed to FigSave.
    lst_ax : list of axis
        list of axis (it is a list even with 1 axis) on which to plot.
    lst_axt : list of axis
        list of secondary axis (it is a list even with 1 axis)
    fig_par : list of float
        ist of parameters to reserve space around the plot canvas.

    """

    if font == 'Times':  # set Times New Roman as the plot font fot text
        # this may require the installation of the font package
        sns.set("paper", font_scale=font_scale)
        sns.set_palette(palette, n_of_colors)
        sns.set_style(sns_style, {'font.family': 'Times New Roman'})
    else:  # leave Dejavu Sans (default) as the plot font fot text
        sns.set("paper", font_scale=font_scale)
        sns.set_palette(palette, n_of_colors)
        sns.set_style(sns_style)
    # single or double column in paperthat the figure will occupy
    if cols > 2:  # numer of columns (thus of plots in the figure)
        raise ValueError('\n FigCreate: cols>2 not supported')

    # width of the figure in inches, it's fixed to keep the same text size
    # is 6, 9, 12 for 1, 1.5, and 3 paper_col (columns in paper)
    fig_wdt = 6*paper_col  # width of the plot in inches
    fig_hgt = 4*paper_col*rows/cols*hgt_mltp  # heigth of the figure in inches
    px = 0.06*(6/fig_wdt)*cols  # set px so that (A) fits the square
    py = px*fig_wdt/fig_hgt/cols*rows/hgt_mltp  # set py so that (A) fits
    # if more rows are added, it increases, but if cols areadded it decreases
    # to maintain the plot ratio
    # set plot margins
    sp_lab_wdt = 0.156/paper_col  # hor. space for labels
    sp_nar_wdt = 0.02294/paper_col  # space narrow no labels (horiz)
    sp_lab_hgt = 0.147/paper_col/rows*cols/hgt_mltp  # space for labels (vert)
    sp_nar_hgt = 0.02/paper_col/rows*cols/hgt_mltp  # space narrow no labels
    # (vert)
    # =========================================================================
    # # 0. Std: standard plot (single or grid rows x cols)
    # =========================================================================
    if plot_type == 0:
        fig, ax = plt.subplots(rows, cols, figsize=(fig_wdt, fig_hgt))
        if rows*cols == 1:  # only 1 plot
            lst_ax = [ax]  # create ax list for uniform iterations over 1 obj.
        elif rows*cols > 1:  # more than one plot
            lst_ax = [axs for axs in ax.flatten()]  # create list of axis
        lst_axt = None  # no secondary axis in this plot_type
        # horizontal space between plot in percentage
        sp_btp_wdt = (0.26*paper_col**2 - 1.09*paper_col + 1.35)
        # vertical space between plot in percentage !!! needs DEBUG
        sp_btp_hgt = .2/paper_col*cols/hgt_mltp
        # left, bottom, right, top, widthspace, heightspace
        fig_par = [sp_lab_wdt, sp_lab_hgt, 1-sp_nar_wdt, 1-sp_nar_hgt,
                   sp_btp_wdt, sp_btp_hgt, px, py]
    # =========================================================================
    # # 1. Twin-x: secondary axis plot (single or grid rows x cols)
    # =========================================================================
    elif plot_type == 1:
        fig, ax = plt.subplots(rows, cols, figsize=(fig_wdt, fig_hgt))
        if rows*cols == 1:  # only 1 plot
            lst_ax = [ax]  # create ax list for uniform iterations over 1 obj.
            lst_axt = [ax.twinx()]  # create a list with secondary axis object
        elif rows*cols > 1:  # more than one plot
            lst_ax = [axs for axs in ax.flatten()]  # create list of axis
            # create list of secondary twin axis
            lst_axt = [axs.twinx() for axs in ax.flatten()]
        # horizontal space between plot in percentage !!! needs DEBUG
        sp_btp_wdt = 1.36*paper_col**2 - 5.28*paper_col + 5.57
        # vertical space between plot in percentage !!! needs DEBUG
        sp_btp_hgt = .2/paper_col*cols/hgt_mltp
        # left, bottom, right(DIFFERENT FROM STD), top, widthspace, heightspace
        fig_par = [sp_lab_wdt, sp_lab_hgt, 1-sp_lab_wdt, 1-sp_nar_hgt,
                   sp_btp_wdt, sp_btp_hgt, px, py]

    return fig, lst_ax, lst_axt, fig_par


def FigSave(fig_name, out_path, fig, lst_ax, lst_axt, fig_par,
            xLab=None, yLab=None, ytLab=None,
            xLim=None, yLim=None, ytLim=None,
            xTicks=None, yTicks=None, ytTicks=None,
            xTickLabels=None, yTickLabels=None, ytTickLabels=None,
            legend=None, ncol_leg=1,
            annotate_lttrs=False, annotate_lttrs_loc='down',
            pdf=False, svg=False, eps=False, transparency=False,
            subfolder=None, tight_layout=False, grid=False, title=False,
            set_size_inches=None
            ):
    '''
    FIXES:
        1. px, py moved to FIgCreate

    This function takes the obects created in FigCreate and allows to modify
    their appeareance and saving the results.

    Parameters
    ----------
    fig_name : str
        name of figure. It is the name of the png od pfd file to be saved
    out_path : pathlib.Path object. path to the output folder.
    fig : figure object. created in FigSave.
    lst_ax : list of axis. Created in FigCreate
    lst_axt : list of twin (secondary) axis. Created in FigCreate
    fig_par : list of figure parameters for space settings
        left, bottom, right, top, widthspace, heightspace, px, py.
        Created in FigCreate
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
    subfolder : str, optional
        name of the subfolder inside the output folder where the output will
        be saved. If the folder does not exists, it is created.
        The default is None.
    '''

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
    vrbls = [xLab, yLab, ytLab, legend]  # collect variables for iteration
    lst_xLab, lst_yLab, lst_ytLab, lst_legend \
        = [], [], [], []  # create lists for iteration
    lst_vrbls = [lst_xLab, lst_yLab, lst_ytLab, lst_legend]  # collect lists
    for vrbl, lst_vrbl in zip(vrbls, lst_vrbls):
        if vrbl is None:  # label is not given for any axis
            lst_vrbl[:] = [None]*n_ax
        else:  # label is given
            if np.size(vrbl) == 1:  # only one value is given
                if type(vrbl) == str:  # create a list before replicating it
                    lst_vrbl[:] = [vrbl]*n_ax  # each axis gets same label
                elif type(vrbl) == list:  # replicate the list
                    lst_vrbl[:] = vrbl*n_ax  # each axis gets same label
            elif np.size(vrbl) == n_ax:  # each axis has been assigned its lab
                lst_vrbl[:] = vrbl  # copy the label inside the list
            else:
                print(vrbl)
                print('Labels/legend size does not match axes number')
    # for xLim, yLim, ytLim creates a list with same length as n_ax.
    # If one list like [a,b] is given, all axis have the same limits, if a list
    # of the same length of the axis is given, each axis has its lim. Where
    # None is given, no lim is set on that axis
    vrbls = [xLim, yLim, ytLim, xTicks, yTicks, ytTicks, xTickLabels,
             yTickLabels, ytTickLabels]  # collect variables for iteration
    lst_xLim, lst_yLim, lst_ytLim, lst_xTicks, lst_yTicks, lst_ytTicks, \
        lst_xTickLabels, lst_yTickLabels, lst_ytTickLabels = \
            [], [], [], [], [], [], [], [], [] # create lists for iteration
    lst_vrbls = [lst_xLim, lst_yLim, lst_ytLim, lst_xTicks, lst_yTicks,
                 lst_ytTicks, lst_xTickLabels, lst_yTickLabels,
                 lst_ytTickLabels]  # collect lists
    for vrbl, lst_vrbl in zip(vrbls, lst_vrbls):
        if vrbl is None:  # limit is not given for any axis
            lst_vrbl[:] = [None]*n_ax
        else:
            # if only list and None are in vrbl, it is [[], None, [], ..]
            # each axis has been assigned its limits
            if any([isinstance(v, (int, float, np.int32, str))
                    for v in vrbl]):
                temporary = []  # necessary to allow append on [:]
                for i in range(n_ax):
                    temporary.append(vrbl)  # give it to all axis
                lst_vrbl[:] = temporary
            else:  # xLim=[[a,b], None, ...] = [list, bool] # no float
                lst_vrbl[:] = vrbl  # a lim for each axis is already given
    # loops over each axs in the ax array and set the different properties
    for i, axs in enumerate(lst_ax):
        # for each property, if the variable is not false, it is set
        if lst_xLab[i] is not None:
            axs.set_xlabel(lst_xLab[i])
        if lst_yLab[i] is not None:
            axs.set_ylabel(lst_yLab[i])
        if lst_xLim[i] is not None:
            axs.set_xlim([lst_xLim[i][0]*(1 + px) - px*lst_xLim[i][1],
                          lst_xLim[i][1]*(1 + px) - px*lst_xLim[i][0]])
        if lst_yLim[i] is not None:
            axs.set_ylim([lst_yLim[i][0]*(1 + py) - py*lst_yLim[i][1],
                          lst_yLim[i][1]*(1 + py) - py*lst_yLim[i][0]])
        if lst_xTicks[i] is not None:
            axs.set_xticks(lst_xTicks[i])
        if lst_yTicks[i] is not None:
            axs.set_yticks(lst_yTicks[i])
        if lst_xTickLabels[i] is not None:
            axs.set_xticklabels(lst_xTickLabels[i])
        if lst_yTickLabels[i] is not None:
            axs.set_yticklabels(lst_yTickLabels[i])
        if grid:
            axs.grid(True)
        if annotate_lttrs is not False:
            if annotate_lttrs_loc == 'down':
                y_lttrs = (py/px*.02)
            elif annotate_lttrs_loc == 'up':
                y_lttrs = 1 - py
            if n_ax == 1:  # if only one plot is given, do not put the letters
                axs.annotate('(' + annotate_lttrs + ')',
                              xycoords='axes fraction',
                              xy=(0, 0), rotation=0, size='large',
                              xytext=(0, y_lttrs), weight='bold')
            elif n_ax > 1:  # if only one plot is given, do not put the letters
                try:  # if specific letters are provided
                    axs.annotate('(' + annotate_lttrs[i] + ')',
                                 xycoords='axes fraction',
                                 xy=(0, 0), rotation=0, size='large',
                                 xytext=(0, y_lttrs), weight='bold')
                except TypeError:  # if no specific letters, use lttrs
                    axs.annotate('(' + lttrs[i] + ')', xycoords='axes fraction',
                                 xy=(0, 0), rotation=0, size='large',
                                 xytext=(0, y_lttrs), weight='bold')

    # if secondary (twin) axis are given, set thier properties
    if lst_axt is not None:
        for i, axst in enumerate(lst_axt):
            axst.grid(False)  # grid is always false on secondaty axis
            # for each property, if the variable is not false, it is set
            if lst_ytLab[i] is not None:
                axst.set_ylabel(lst_ytLab[i])
            if lst_ytLim[i] is not None:
                axst.set_ylim([lst_ytLim[i][0]*(1 + py) - py*lst_ytLim[i][1],
                              lst_ytLim[i][1]*(1 + py) - py*lst_ytLim[i][0]])
            if lst_ytTicks[i] is not None:
                axst.set_yticks(lst_ytTicks[i])
            if lst_ytTickLabels[i] is not None:
                axst.set_yticklabels(lst_ytTickLabels[i])
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
                axs.legend(hnd_ax + hnd_axt, lab_ax + lab_axt, loc=lst_legend[i],
                           ncol=ncol_leg)
                i += 1
    try:
        fig.align_labels()  # align labels of subplots, needed only for multi plot
    except AttributeError:
        print('align_labels not performed')
    # if a subfolder is specified, create the subfolder inside the output
    # folder if not already there and save the figure in it
    if subfolder is not None:
        out_path = plib.Path(out_path, subfolder)  # update out_path
        plib.Path(out_path).mkdir(parents=True, exist_ok=True)  # check if
        # folder is there, if not create it
    # set figure margins and save the figure in the output folder
    if set_size_inches:
        fig.set_size_inches(set_size_inches)
    if tight_layout is False:  # if margins are given sets margins and save
        fig.subplots_adjust(*fig_adj_par[0:6])  # set margins
        plt.savefig(plib.Path(out_path, fig_name + '.png'), dpi=300,
                    transparent=transparency)
        if pdf is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, fig_name + '.pdf'))
        if svg is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, fig_name + '.svg'))
        if eps is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, fig_name + '.eps'))
    else:  # margins are not given, use a tight layout option and save
        plt.savefig(plib.Path(out_path, fig_name + '.png'),
                    bbox_inches="tight", dpi=300, transparent=transparency)
        if pdf is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, fig_name + '.pdf'),
                        bbox_inches="tight")
        if svg is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, fig_name + '.svg'),
                        bbox_inches="tight")
        if eps is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, fig_name + '.eps'),
                        bbox_inches="tight")
    # add the title after saving, so it's only visible in the console
    if title is True:
        lst_ax[0].annotate(fig_name, xycoords='axes fraction', size='small',
                            xy=(0, 0), xytext=(0.05, .95), clip_on=True)
    return

# =============================================================================
# # TGA functions and classes
# =============================================================================

def multi_test(out_path, exps, paper_col=.78, subfolder=None, fig_name='Multi',
               labels=None,
               xLim=[115, 895], xTicks=None,
               ProxCombReport=True,
               plt_tg=False, yLim_tg=[0, 100], yTicks_tg=None,
               plt_dtg=False, yLim_dtg=None, yTicks_dtg=None, dtg_hgt_mltp=1.25,
               plt_tgdtg=True, lttrs_tg=False, lttrs_dtg=False, lttrs_tgdtg=True,
               tgdtg_hgt_mltp=1.25,
               plt_cscd=False, clrs_cscd=False,
               yLim0_cscd=-8, shifts_cscd=np.asarray([0, 11, 5, 10, 10, 10, 11]),
               peaks_cscd=None, peak_names=None, dh_names_cscd=0.1, loc_names_cscd=130,
               hgt_mltp_cscd=1.5, legend_cscd='lower right', y_values_cscd=[-10, 0],
               letter_cscd=False,
               plt_gc=False, gc_lim=300, gc_dt=None, gc_hgt_mltp=1.25,
               lttrs_gc=False,
               pdf=False, svg=False, eps=False, TG_lab='TG [wt%]',
               DTG_lab='DTG [wt%/min]', grid=False):
    # create folder if not existent
    plib.Path(out_path, 'MultiSamples').mkdir(parents=True, exist_ok=True)
    out_path_MS = plib.Path(out_path, 'MultiSamples')
    if not labels:  # try with labels and use name if no label is given
        labels = [exp.label if exp.label else exp.name for exp in exps]
    if plt_tg:
        fig, ax, axt, fig_par = FigCreate(rows=1, cols=1, plot_type=0,
                                          paper_col=paper_col,
                                          hgt_mltp=dtg_hgt_mltp)
        for i, exp in enumerate(exps):
            ax[0].plot(exp.T, exp.mp_db, color=clrs[i],
                       linestyle=lnstls[i], label=labels[i])
            ax[0].fill_between(exp.T, exp.mp_db - exp.mp_db_std,
                               exp.mp_db + exp.mp_db_std, color=clrs[i],
                               alpha=.3)
            # ax[0].plot(exp.T_dtg, exp.mp_db_dtg, color=clrs[i],
            #            linestyle=lnstls[i], label=labels[i])
            # ax[0].fill_between(exp.T_dtg, exp.mp_db_dtg - exp.mp_db_dtg_std,
            #                    exp.mp_db_dtg + exp.mp_db_dtg_std, color=clrs[i],
            #                    alpha=.3)
        FigSave(fig_name + '_tg', out_path_MS, fig, ax, axt, fig_par,
                subfolder=subfolder, xLim=xLim, yLim=yLim_tg,
                yTicks=yTicks_tg,
                xLab=exps[0].T_unit, legend='upper right',
                yLab=TG_lab, annotate_lttrs=lttrs_tg, grid=grid)
    if plt_dtg:
        fig, ax, axt, fig_par = FigCreate(rows=1, cols=1, plot_type=0,
                                          paper_col=paper_col,
                                          hgt_mltp=dtg_hgt_mltp)
        for i, exp in enumerate(exps):
            ax[0].plot(exp.T_dtg, exp.dtg_db, color=clrs[i],
                       linestyle=lnstls[i], label=labels[i])
            ax[0].fill_between(exp.T_dtg, exp.dtg_db - exp.dtg_db_std,
                               exp.dtg_db + exp.dtg_db_std, color=clrs[i],
                               alpha=.3)
        if plt_gc:
            ax[0].vlines(gc_lim, ymin=yLim_dtg[0], ymax=yLim_dtg[1],
                         linestyle=lnstls[1], color=clrs[7],
                         label='T$_{max GC-MS}$')
        ax[0].legend(loc='lower right')
        FigSave(fig_name + '_dtg', out_path_MS, fig, ax, axt, fig_par,
                xLim=xLim, yLim=yLim_dtg,
                yTicks=yTicks_dtg,
                yLab=DTG_lab, xLab=exps[0].T_unit,
                pdf=pdf, svg=svg, annotate_lttrs=lttrs_dtg, grid=grid)
    if plt_tgdtg:
        fig, ax, axt, fig_par = FigCreate(rows=2, cols=1, plot_type=0,
                                          paper_col=paper_col,
                                          hgt_mltp=tgdtg_hgt_mltp)
        for i, exp in enumerate(exps):
            ax[0].plot(exp.T, exp.mp_db, color=clrs[i],
                       linestyle=lnstls[i], label=labels[i])
            ax[0].fill_between(exp.T, exp.mp_db - exp.mp_db_std,
                               exp.mp_db + exp.mp_db_std, color=clrs[i],
                               alpha=.3)
            # ax[0].plot(exp.T_dtg, exp.mp_db_dtg, color=clrs[i],
            #            linestyle=lnstls[i], label=labels[i])
            # ax[0].fill_between(exp.T_dtg, exp.mp_db_dtg - exp.mp_db_dtg_std,
            #                    exp.mp_db_dtg + exp.mp_db_dtg_std, color=clrs[i],
            #                    alpha=.3)
            ax[1].plot(exp.T_dtg, exp.dtg_db, color=clrs[i],
                       linestyle=lnstls[i], label=labels[i])
            ax[1].fill_between(exp.T_dtg, exp.dtg_db - exp.dtg_db_std,
                               exp.dtg_db + exp.dtg_db_std, color=clrs[i],
                               alpha=.3)
        if plt_gc:
            for i, exp in enumerate(exps):
                T_gc = exp.T_dtg[exp.T_dtg > gc_lim][0]
                mp_gc =  exp.mp_db_dtg[exp.T_dtg > gc_lim][0]
                ax[0].hlines(mp_gc, xmin=xLim[0] + 10,
                             xmax=T_gc, color=clrs[i], linestyle=lnstls[i])
                text = str(int(100 - mp_gc)) + '%'
                dt = i*15
                if gc_dt is not None:
                    dt = gc_dt[i]
                ax[0].annotate(text, ha='left', va='bottom',
                               xy=(xLim[0] + dt, mp_gc -.3), fontsize=9,
                               color=clrs[i])
            ax[0].vlines(gc_lim, ymin=0, ymax=100,
                         linestyle=lnstls[1], color=clrs[7],
                         label='T$_{max GC-MS}$')
        if lttrs_tg is not None and lttrs_dtg is not None:
            lttrs_tgdtg = [lttrs_tg, lttrs_dtg]
        FigSave(fig_name + '_tgdtg', out_path_MS, fig, ax, axt, fig_par,
                xLim=xLim, yLim=[yLim_tg, yLim_dtg],
                legend=['upper right', 'lower right'],
                xLab=exps[0].T_unit, yLab=[TG_lab, DTG_lab],
                yTicks=[yTicks_tg, yTicks_dtg],
                annotate_lttrs=lttrs_tgdtg, svg=svg, pdf=pdf, eps=eps, grid=grid)
    if plt_cscd:
        yLim_cscd=[yLim0_cscd, np.sum(shifts_cscd)]
        dh = np.cumsum(shifts_cscd)
        fig, ax, axt, fig_par = FigCreate(1, 1, paper_col=.78,
                                          hgt_mltp=hgt_mltp_cscd)
        for n, exp in enumerate(exps):
            if clrs_cscd:
                ax[0].plot(exp.T_dtg, exp.dtg_db + dh[n], color=clrs[n],
                           linestyle=lnstls[0])
                ax[0].fill_between(exp.T_fit, exp.dtg_db - exp.dtg_db_std
                                   + dh[n], exp.dtg_db + exp.dtg_db_std + dh[n],
                                   color=clrs[n], alpha=.3)
            else:
                ax[0].plot(exp.T_dtg, exp.dtg_db + dh[n], color='k',
                           linestyle=lnstls[0])
                ax[0].fill_between(exp.T_dtg, exp.dtg_db - exp.dtg_db_std
                                   + dh[n], exp.dtg_db + exp.dtg_db_std + dh[n],
                                   color='k', alpha=.3)
            ax[0].annotate(labels[n], ha='left', va='bottom',
                           xy=(loc_names_cscd,
                               exp.dtg_db[np.argmax(exp.T_dtg>
                                                     loc_names_cscd)] +
                                          dh[n] + dh_names_cscd))
        if peaks_cscd:
            for p, peak in enumerate(peaks_cscd):
                if peak:  # to allow to use same markers by skipping peaks
                    ax[0].plot(peak[0], peak[1], linestyle='None',
                               marker=mrkrs[p], color='k', label=peak_names[p])
        if y_values_cscd:
            ax[0].set_yticks(y_values_cscd)
        else:
            ax[0].set_yticks([])
        FigSave(fig_name + '_cscd', out_path_MS, fig, ax, axt, fig_par,
                legend=legend_cscd, annotate_lttrs=letter_cscd,
                xLab=exps[0].T_unit, yLab=DTG_lab,
                xLim=xLim, yLim=yLim_cscd, svg=svg, pdf=pdf, eps=eps)
    if ProxCombReport:
        Reports = pd.DataFrame(columns=list(exps[0].ProxComb))
        for i, exp in enumerate(exps):
            Reports.loc[exp.label + '_ave'] = exp.ProxComb.loc['average', :]
        for i, exp in enumerate(exps):
            Reports.loc[exp.label + '_std'] = exp.ProxComb.loc['std', :]
        Reports.to_excel(plib.Path(out_path_MS, fig_name + '_ProxComb.xlsx'))
        return Reports


def solid_distill(out_path, exps, subfolder='',
                  steps=[40, 70, 100, 130, 160, 190], plt_ar=True, plt_db=True,
                  TG_lab='TG [wt%]', DTG_lab='DTG [wt%/min]', grid=False,
                  hgt_mltp=1.25, paper_col=.78, labels=None, fig_name="Dist",
                  xLim=None, yLim_ar=None, yTicks_ar=None, lttrs_ar=None,
                  legend=['upper left', 'upper right'], print_dfs=True):
    """
    produces tg and dtg plots of each replicate in a sample. Quality of plots
    is supposed to allow checking for errors.

    """
    plib.Path(out_path, 'SolidDistill').mkdir(parents=True, exist_ok=True)
    out_path_MS = plib.Path(out_path, 'SolidDistill')
    if not labels:  # try with labels and use name if no label is given
        labels = [exp.label if exp.label else exp.name for exp in exps]
    df_ar = pd.DataFrame(index=labels, columns=steps+['end'])
    df_db = pd.DataFrame(index=labels, columns=steps+['end'])
    for i, exp in enumerate(exps):
        idxs = []
        for step in steps:
            idxs.append(np.argmax(exp.time > step))
        exp.idxs_dist_steps = idxs.append(len(exp.time)-1)
        exp.T_dist = exp.T[idxs]
        exp.time_dist = exp.time[idxs]

        exp.dmp_ar_dist = -np.diff(exp.mp_ar[idxs], prepend=100)
        exp.dmp_db_dist =  -np.diff(exp.mp_db[idxs], prepend=100)

        exp.loc_ar_dist = np.convolve(np.insert(exp.mp_ar[idxs], 0, 100),
                                     [.5, .5], mode='valid')
        exp.loc_db_dist = np.convolve(np.insert(exp.mp_db[idxs], 0, 100),
                                     [.5, .5], mode='valid')
        df_ar.iloc[i, :] = exp.loc_ar_dist
        df_db.iloc[i, :] = exp.loc_db_dist
    if print_dfs:
        df_ar.to_excel(plib.Path(out_path_MS, fig_name + '_ar.xlsx'))
        df_db.to_excel(plib.Path(out_path_MS, fig_name + '_db.xlsx'))
    if plt_ar:
        fig, ax, axt, fig_par = FigCreate(rows=2, cols=1, plot_type=0,
                                          paper_col=paper_col,
                                          hgt_mltp=hgt_mltp)
        for i, exp in enumerate(exps):


            ax[0].plot(exp.time, exp.T, color=clrs[i],
                       linestyle=lnstls[i], label=labels[i])
            # ax[0].plot(exp.time, np.diff(exp.T, prepend=np.nan)*100, color=clrs[i],
            #            linestyle=lnstls[i], label=labels[i])
            ax[0].fill_between(exp.time, exp.T - exp.T_std,
                               exp.T + exp.T_std, color=clrs[i],
                               alpha=.3)
            ax[1].plot(exp.time, exp.mp_ar, color=clrs[i],
                       linestyle=lnstls[i], label=labels[i])
            # ax[1].plot(exp.time, np.diff(exp.mp_ar, prepend=np.nan)*100, color=clrs[i],
            #            linestyle=lnstls[i], label=labels[i])
            ax[1].fill_between(exp.time, exp.mp_ar - exp.mp_ar_std,
                               exp.mp_ar + exp.mp_ar_std, color=clrs[i],
                               alpha=.3)
            for tm, mp, dmp in zip(exp.time_dist, exp.loc_db_dist,
                                   exp.dmp_ar_dist):
                ax[1].annotate(str(np.round(dmp, 0)) + '%',
                               ha='center', va='top',
                               xy=(tm - 10, mp+1), fontsize=9, color=clrs[i])
            ax[0].legend(loc='upper left')
        FigSave(fig_name + 'dist_ar', out_path_MS, fig, ax, axt, fig_par,
                subfolder=subfolder, xLim=xLim, yLim=yLim_ar,
                yTicks=yTicks_ar,
                xLab='time [min]', legend=None,
                yLab=[exps[0].T_unit, TG_lab+'(stb)'],
                annotate_lttrs=lttrs_ar, grid=grid)
    if plt_db:
        fig, ax, axt, fig_par = FigCreate(rows=2, cols=1, plot_type=0,
                                          paper_col=paper_col,
                                          hgt_mltp=hgt_mltp)
        for i, exp in enumerate(exps):


            ax[0].plot(exp.time, exp.T, color=clrs[i],
                       linestyle=lnstls[i], label=labels[i])
            # ax[0].plot(exp.time, np.diff(exp.T, prepend=np.nan)*100, color=clrs[i],
            #            linestyle=lnstls[i], label=labels[i])
            ax[0].fill_between(exp.time, exp.T - exp.T_std,
                               exp.T + exp.T_std, color=clrs[i],
                               alpha=.3)
            ax[1].plot(exp.time, exp.mp_db, color=clrs[i],
                       linestyle=lnstls[i], label=labels[i])
            ax[1].fill_between(exp.time, exp.mp_db - exp.mp_db_std,
                               exp.mp_db + exp.mp_db_std, color=clrs[i],
                               alpha=.3)
            for tm, mp, dmp in zip(exp.time_dist, exp.loc_db_dist,
                                   exp.dmp_db_dist):
                ax[1].annotate(str(np.round(dmp, 0)) + '%',
                               ha='center', va='top',
                               xy=(tm - 10, mp+1), fontsize=9, color=clrs[i])
            ax[0].legend(loc='upper left')
        FigSave(fig_name + 'dist_db', out_path_MS, fig, ax, axt, fig_par,
                subfolder=subfolder, xLim=xLim, yLim=yLim_ar,
                yTicks=yTicks_ar,
                xLab='time [min]',
                yLab=[exps[0].T_unit, TG_lab+'(db)'],
                annotate_lttrs=lttrs_ar, grid=grid)
    return df_ar, df_db


def KASkinetics(out_path, exps_all, rmps, subfolder='KASkinetics',
                alphas=np.arange(0.05, .9, 0.05), labels_exps=None,
                fig_name='CAS', xLim_eatt=[0, 1], yLim_eatt=None,
                xLim_iso=None, yLim_iso=None,
                yTicks_Eatt=None, dx_leg=0, dy_leg=0, annt_names=True,
                paper_col_isolines=1.2, leg_cols=1, paper_col_Ea=1.2,
                grid=False, plt_type_Ea='scatter'):
        # for plots
        labels_exps = []
        R_gas = 8.314462618
        n_exp = len(exps_all)  # number of feedstocks
        n_rmps = len(rmps)  # number of ramp used for feedstock
        n_alpha = len(alphas)  # number of alpha investigated
        xmatrs, ymatrs, v_fits, Eas, Eas_std = [], [], [], [], []
        ######################

        v_a = np.zeros(n_alpha, dtype=int)
        for ee, exps_ramp in enumerate(exps_all):
            labels_exps.append(exps_ramp[0].label.split('-')[0])
            Ea = np.zeros(n_alpha)
            Ea_std = np.zeros(n_alpha)
            v_fit = []
            v_res_fit = np.zeros(n_alpha)
            r_sqrd_fit = np.zeros(n_alpha)
            ymatr = np.zeros((n_alpha, n_rmps))
            xmatr = np.zeros((n_alpha, n_rmps))
            for e, exp in enumerate(exps_ramp):
                # build the two matrixes
                if exp.T_unit == 'T [K]':
                    TK = exp.T_dtg
                elif exp.T_unit == 'T [°C]':
                    TK = exp.T_dtg + 273.15
                mp_db_dtg = exp.mp_db_dtg
                mdaf = ((mp_db_dtg - np.min(mp_db_dtg))
                        / (np.max(mp_db_dtg) - np.min(mp_db_dtg)))
                a = 1 - mdaf
                for i in range(n_alpha):
                    v_a[i] = np.argmax(a > alphas[i])
                xmatr[:, e] = 1/TK[v_a]*1000
                for i in range(n_alpha):
                    # BETA IS IN MINUTE HERE
                    ymatr[i, e] = np.log(rmps[e]/TK[v_a[i]]**2)

            for i in range(n_alpha):
                p, cov = np.polyfit(xmatr[i, :], ymatr[i, :], 1, cov=True)
                v_fit.append(np.poly1d(p))
                v_res_fit = ymatr[i, :] - v_fit[i](xmatr[i, :])
                r_sqrd_fit[i] = (1 - (np.sum(v_res_fit**2)
                                      / np.sum((ymatr[i, :]
                                                - np.mean(ymatr[i, :]))**2)))
                Ea[i] = -v_fit[i][1]* R_gas
                Ea_std[i] = np.sqrt(cov[0][0])* R_gas
            xmatrs.append(xmatr)
            ymatrs.append(ymatr)
            v_fits.append(v_fit)
            Eas.append(Ea)
            Eas_std.append(Ea_std)
        # plot isolines
        if n_exp <= 3:
            fig, ax, axt, fig_par = FigCreate(rows=n_exp, cols=1, plot_type=0,
                                              paper_col=paper_col_isolines)
        else:
            fig, ax, axt, fig_par = FigCreate(rows=n_exp//2, cols=2,
                                              plot_type=0,
                                              paper_col=1.5)
        for ee, exps_ramp in enumerate(exps_all):
            ymaxiso = np.max(ymatrs[ee])
            yminiso = np.min(ymatrs[ee])
            if n_exp > 1:
                annotate_lttrs_iso = True
            else:
                annotate_lttrs_iso = False
            for i in range(n_alpha):
                lab = r'$\alpha$=' + str(np.round(alphas[i], 2))
                if xLim_iso:
                    x=np.linspace(xLim_iso[0], xLim_iso[1], 200)
                xmin = np.argwhere(v_fits[ee][i](x)<ymaxiso)[0][0]
                try:
                    xmax = np.argwhere(v_fits[ee][i](x)<yminiso)[0][0]
                except IndexError:
                    xmax = 0
                newx = x[xmin:xmax]
                ax[ee].plot(newx, v_fits[ee][i](newx),
                            color=clrs[i], linestyle=lnstls[i])
                ax[ee].plot(xmatrs[ee][i, :], ymatrs[ee][i, :], color=clrs[i],
                            linestyle='None', marker=mrkrs[i])
                ax[ee].plot([], [], color=clrs[i], linestyle=lnstls[i],
                            marker=mrkrs[i], label=lab)
                hnd_ax, lab_ax = ax[ee].get_legend_handles_labels()
            if annt_names:
                ax[ee].annotate(exps_ramp[0].name[:-3], xycoords="axes fraction",
                                xy=(0, 0), rotation=0, size="small",
                                xytext=(0.05, 0.93),)
        if n_exp == 1:
            ax[0].legend(ncol=leg_cols)
        elif n_exp <= 3:
            ax[0].legend(bbox_to_anchor=(1.0 + dx_leg, 1.03 + dy_leg))
        else:
            ax[1].legend(bbox_to_anchor=(1.0 + dx_leg, 1.03 + dy_leg))
        FigSave('IsoLines' + fig_name, out_path, fig, ax, axt, fig_par,
                subfolder=subfolder, xLim=xLim_iso, yLim=yLim_iso,
                xLab='1000/T [1/K]', legend=None,
                annotate_lttrs=annotate_lttrs_iso,
                yLab=r'ln($\beta$/T$^{2}$)', tight_layout=False)
        # plot activation energy
        fig, ax, axt, fig_par = FigCreate(rows=1, cols=1, plot_type=0,
                                          paper_col=paper_col_Ea)
        for ee, exps_ramp in enumerate(exps_all):
            if plt_type_Ea=='scatter':
                ax[0].errorbar(alphas, Eas[ee], Eas_std[ee], color='k',
                               linestyle='None', capsize=3)
                print(labels_exps)
                ax[0].plot(alphas, Eas[ee], color=clrs[ee],
                           linestyle=lnstls[ee], marker=mrkrs[ee],
                           label=labels_exps[ee])
            elif plt_type_Ea=='bar':

                ax[0].bar(alphas, Eas[ee], color=clrs[ee],
                          label=labels_exps[ee], width=.04,
                          edgecolor='k')
                ax[0].errorbar(alphas, Eas[ee], Eas_std[ee], color='k',
                               linestyle='None', capsize=3)
            elif plt_type_Ea=='line':

                ax[0].plot(alphas, Eas[ee], color=clrs[ee],
                           linestyle=lnstls[ee],
                           label=labels_exps[ee])
                ax[0].fill_between(alphas, Eas[ee] - Eas_std[ee],
                                  Eas[ee] + Eas_std[ee], color=clrs[ee],
                                  alpha=.3)
        if n_exp == 1:
            legend_loc = False
        else:
            legend_loc = 'best'
        FigSave('Eatt' + fig_name + '_' + plt_type_Ea, out_path, fig, ax, axt, fig_par,
                subfolder=subfolder, xLim=xLim_eatt, yLim=yLim_eatt,
                legend=legend_loc,
                yTicks=yTicks_Eatt, xLab=r'$\alpha$ [-]',
                yLab=r'$E_{a}$ [kJ/mol]', grid=grid)




class peak:
    def __init__(self, name=None, modelcode='G', center=0, amplitude=200,
                 sigma=20, gamma=0, c_min=None, c_max=None,
                 label=None, a_min=None, a_max=None, s_min=None, s_max=None,):
        self.name = name
        self.modelcode = modelcode  # 'G'==gaussian
        if modelcode == 'G' or modelcode == 'g':
            self.modeltype = GaussianModel  # 'G'==gaussian

        self.center = center
        self.amplitude = amplitude
        self.sigma = sigma
        self.gamma = gamma
        if label:
            self.label = label
        else:
            self.label = modelcode + '-' + name
        self.c_min = c_min
        self.c_max = c_max
        self.a_min = a_min
        self.a_max = a_max
        self.s_min = s_min
        self.s_max = s_max


def deconvolute_tga(
    out_path,
    exps,
    fig_name="dcv",
    recompute=False,
    subfolder='Deconvolute',
    plt_single=True,
    plt_all=True,
    print_mtx_dcv=False,
    plt_sign=-1,
    TLim=None,
    yLim=None,
    yLim_single=None,
    xLim=None,
    paper_col=0.78,
    labels=None,
    yTicks1=None,
    yTicks2=None,
    pdf=False,
    svg=False,
    plt_single_all=True,
    plt_peaks_all=True,
    yLim_peaks=None,
    ytLim_peaks=None,
    x_lab_rot_peak=0,
    legend_peaks="best",
    lttrs_single=False,
):
    print("deconvolute_dsc function called")
    # n_peaks must be the same in all exps
    n_peaks = len(exps[0].first_guess_peaks)
    if plt_sign == -1:  # to print curves with their sign
        ps = -1
        legend_loc = "lower right"  # if curves are downwards, is best location
    elif plt_sign == 1:  # print curves upwards
        ps = 1
        legend_loc = "upper right"  # if curves are upwards, is best location
    if not labels:
        labels = []
        for exp in exps:
            if exp.label:
                labels.append(exp.label)
            else:
                labels.append(exp.name)
    for e, exp in enumerate(exps):
        if recompute:  # if recompute is True it always does the computation
            pass
        else:  # if recompute is false, it checks if the compute was done
            if exp.debug_dcv_done:  # if the computation is already done
                print(exp.name, "skipped")
                continue  # goes to next exp
        print(exp.name, "doing")  # does the dcvnlt
        idx_xLim0 = np.argmax(exp.T_dtg > TLim[0])  # computation limits
        idx_xLim1 = np.argmax(exp.T_dtg > TLim[1])
        exp.dcv_x = exp.T_dtg[idx_xLim0:idx_xLim1]  # x is the same in runs
        len_v = int(idx_xLim1 - idx_xLim0)  # len of arrays
        # np arrays creation
        exp.dcv_peak_names = []  # peaks have to be the same for all runs
        exp.dcv_T_stk = np.zeros((exp.n_repl, len_v))
        exp.dcv_y_stk = np.zeros((exp.n_repl, len_v))
        exp.dcv_best_fit_stk = np.zeros((exp.n_repl, len_v))
        exp.dcv_r2_stk = np.zeros(exp.n_repl)
        exp.dcv_peaks_stk = np.zeros((exp.n_repl, n_peaks, len_v))
        exp.dcv_peak_areas_stk = np.zeros((exp.n_repl, n_peaks))
        exp.dcv_peak_maxs_stk = np.zeros((exp.n_repl, n_peaks))
        exp.dcv_peak_amps_stk = np.zeros((exp.n_repl, n_peaks))
        exp.dcv_peak_centers_stk = np.zeros((exp.n_repl, n_peaks))
        exp.dcv_peak_sigmas_stk = np.zeros((exp.n_repl, n_peaks))

        for r in range(exp.n_repl):  # r iterates over replicate runs
            # dcv_T and _y are taken for each replicate
            # exp.dcv_T_stk[r, :] = exp.T_stk[r, idx_xLim0:idx_xLim1]
            exp.dcv_y_stk[r, :] = abs(exp.dtg_db_stk[r, idx_xLim0:idx_xLim1])
            model = LinearModel(prefix="bkg_")  # used only to initialize model
            params = model.make_params()
            params["bkg_intercept"].set(0, vary=False)  # set bkg to 0 amplit
            params["bkg_slope"].set(0, vary=False)  # set bkg to 0 slope
            # for each peak add it to model and set given params
            for i, peak in enumerate(exp.first_guess_peaks):
                model_add = peak.modeltype(prefix=peak.name)
                pars = model_add.make_params()
                pars[peak.name + "center"].set(
                    peak.center, min=peak.c_min, max=peak.c_max
                )
                pars[peak.name + "sigma"].set(
                    peak.sigma, min=peak.s_min, max=peak.s_max
                )
                pars[peak.name + "amplitude"].set(
                    peak.amplitude, min=peak.a_min, max=peak.a_max
                )
                try:  # some Models do not have gamma param
                    pars[peak.name + "gamma"].set(
                        peak.gamma, min=peak.g_min, max=peak.g_max
                    )
                    pars[peak.name + "c"].set(50)
                except KeyError:
                    pass  # the chosen curve has no gamma parameter, skip this
                model = model + model_add
                params.update(pars)
            # init = model.eval(params, x=x)  # results with initial guesses
            result = model.fit(exp.dcv_y_stk[r, :], params, x=exp.dcv_x)
            best_values = result.best_values  # best fit
            r2 = (1 - result.residual.var() /
                  np.var(exp.dcv_y_stk[r, :])) * 100
            comps = result.eval_components()  # single peaks
            exp.dcv_best_fit_stk[r, :] = result.best_fit  # store best fit
            exp.dcv_r2_stk[r] = r2  # store r2
            p = 0  # counts the peaks
            for name, comp in comps.items():
                if name == "bkg_":  # bkg is always zero, is skipped
                    continue  # starts from next iteration of the outer if
                exp.dcv_peaks_stk[r, p, :] = comp  # peak array
                exp.dcv_peak_names.append(exp.first_guess_peaks[p].name)
                exp.dcv_peak_centers_stk[r, p] = best_values[name + "center"]
                exp.dcv_peak_areas_stk[r, p] = ps * np.sum(comp)
                exp.dcv_peak_maxs_stk[r, p] = ps * np.max(abs(comp))
                exp.dcv_peak_amps_stk[r, p] = best_values[name + "amplitude"]
                p += 1
            print("run", str(r), ": R2 = ", exp.dcv_r2_stk[r].round(2))
        # average values and std dev for each exp (over replicates r)
        exp.dcv_T = np.average(exp.dcv_T_stk, axis=0)
        exp.dcv_T_std = np.std(exp.dcv_T_stk, axis=0)
        exp.dcv_y = np.average(exp.dcv_y_stk, axis=0)
        exp.dcv_y_std = np.std(exp.dcv_y_stk, axis=0)
        exp.dcv_best_fit = np.average(exp.dcv_best_fit_stk, axis=0)
        exp.dcv_best_fit_std = np.std(exp.dcv_best_fit_stk, axis=0)
        exp.dcv_r2 = np.average(exp.dcv_r2_stk)
        exp.dcv_r2_std = np.std(exp.dcv_r2_stk)
        exp.dcv_peaks = np.average(exp.dcv_peaks_stk, axis=0)

        exp.dcv_peaks_std = np.std(exp.dcv_peaks_stk, axis=0)
        exp.dcv_peak_areas = np.average(exp.dcv_peak_areas_stk, axis=0)
        exp.dcv_peak_areas_std = np.std(exp.dcv_peak_areas_stk, axis=0)
        exp.dcv_peak_maxs = np.average(exp.dcv_peak_maxs_stk, axis=0)
        exp.dcv_peak_maxs_std = np.std(exp.dcv_peak_maxs_stk, axis=0)
        exp.dcv_peak_amps = np.average(exp.dcv_peak_amps_stk, axis=0)
        exp.dcv_peak_amps_std = np.std(exp.dcv_peak_amps_stk, axis=0)
        exp.dcv_peak_centers = np.average(exp.dcv_peak_centers_stk, axis=0)
        exp.dcv_peak_centers_std = np.std(exp.dcv_peak_centers_stk, axis=0)
        exp.dcv_peak_sigmas = np.average(exp.dcv_peak_sigmas_stk, axis=0)
        exp.dcv_peak_sigmas_std = np.std(exp.dcv_peak_sigmas_stk, axis=0)
        exp.debug_dcv_done = True  # to avoid recalculations
    if plt_single:  # a single plot for each exp dcv is created\
        subfolder_single = subfolder + "\Single"
        for e, exp in enumerate(exps):

            fig, ax, axt, fig_par = FigCreate(
                rows=1, cols=1, plot_type=0, paper_col=0.78
            )
            ax[0].plot(
                exp.dcv_x,
                exp.dcv_y * ps,
                color="k",
                linestyle=lnstls[0],
                linewidth=1.5,
                label="Exp. (" + labels[e] + ")",
            )
            ax[0].fill_between(
                exp.dcv_x,
                exp.dcv_y * ps - exp.dcv_y_std,
                exp.dcv_y * ps + exp.dcv_y_std,
                color="k",
                alpha=0.3,
            )
            ax[0].plot(
                exp.dcv_x,
                exp.dcv_best_fit * ps,
                color="r",
                linestyle=lnstls[0],
                linewidth=1.5,
                label="Best fit",
            )
            ax[0].fill_between(
                exp.dcv_x,
                exp.dcv_best_fit * ps - exp.dcv_best_fit_std,
                exp.dcv_best_fit * ps + exp.dcv_best_fit_std,
                color="r",
                alpha=0.3,
            )
            ax[0].annotate(
                "R$^2$="
                + str(exp.dcv_r2.round(1))
                + "$\pm$"
                + str(exp.dcv_r2_std.round(1)),
                xycoords="axes fraction",
                xy=(0, 0),
                rotation=0,
                size="small",
                xytext=(0.75, 0.93),
            )
            for p in range(n_peaks):
                col = 0  # avoid color[3], red, which is used for best fit
                if p >= 3:
                    col = 1
                ax[0].plot(
                    exp.dcv_x,
                    exp.dcv_peaks[p, :] * ps,
                    color=clrs[p + col],
                    linestyle=lnstls[1 + p],
                    label=exp.first_guess_peaks[p].label,
                )
                ax[0].fill_between(
                    exp.dcv_x,
                    exp.dcv_peaks[p, :] * ps - exp.dcv_peaks_std[p, :],
                    exp.dcv_peaks[p, :] * ps + exp.dcv_peaks_std[p, :],
                    color=clrs[p + col],
                    alpha=0.3,
                )
            if lttrs_single:
                lttrs_plot = lttrs[e]
            else:
                lttrs_plot = False

            FigSave(
                exp.name + "_dcv",
                out_path,
                fig,
                ax,
                axt,
                fig_par,
                subfolder=subfolder_single,
                yLim=yLim_single,
                xLim=xLim,
                xLab="time [min]",
                legend=legend_loc,
                yLab='DTG [% min$^{-1}$]',
                annotate_lttrs=lttrs_plot
            )
    if plt_single_all:  # single plots but in a single figure shell
        fig, ax, axt, fig_par = FigCreate(
            rows=len(exps), cols=1, plot_type=0, paper_col=0.78
        )
        for e, exp in enumerate(exps):
            ax[e].plot(
                exp.dcv_x,
                exp.dcv_y * ps,
                color="k",
                linestyle=lnstls[0],
                linewidth=1.5,
                label="Exp. (" + labels[e] + ")",
            )
            ax[e].fill_between(
                exp.dcv_x,
                exp.dcv_y * ps - exp.dcv_y_std,
                exp.dcv_y * ps + exp.dcv_y_std,
                color="k",
                alpha=0.3,
            )
            ax[e].plot(
                exp.dcv_x,
                exp.dcv_best_fit * ps,
                color="r",
                linestyle=lnstls[0],
                linewidth=1.5,
                label="Best fit",
            )
            ax[e].fill_between(
                exp.dcv_x,
                exp.dcv_best_fit * ps - exp.dcv_best_fit_std,
                exp.dcv_best_fit * ps + exp.dcv_best_fit_std,
                color="r",
                alpha=0.3,
            )
            ax[e].annotate(
                "R$^2$="
                + str(exp.dcv_r2.round(1))
                + "$\pm$"
                + str(exp.dcv_r2_std.round(1)),
                xycoords="axes fraction",
                xy=(0, 0),
                rotation=0,
                size="small",
                xytext=(0.75, 0.93),
            )
            for p in range(n_peaks):
                col = 0
                if p >= 3:
                    col = 1
                ax[e].plot(
                    exp.dcv_x,
                    exp.dcv_peaks[p, :] * ps,
                    color=clrs[p + col],
                    linestyle=lnstls[1 + p],
                    label=exp.first_guess_peaks[p].label,
                )
                ax[e].fill_between(
                    exp.dcv_x,
                    exp.dcv_peaks[p, :] * ps - exp.dcv_peaks_std[p, :],
                    exp.dcv_peaks[p, :] * ps + exp.dcv_peaks_std[p, :],
                    color=clrs[p + col],
                    alpha=0.3,
                )
        FigSave(
            fig_name + "_dcvAll",
            out_path,
            fig,
            ax,
            axt,
            fig_par,
            subfolder=subfolder,
            yLim=yLim_single,
            xLim=xLim,
            xLab='T [°C]',
            legend=legend_loc,
            annotate_lttrs=True,
            yLab='DTG [% min$^{-1}$]',
        )
    if plt_peaks_all:
        rows = n_peaks
        cols = 1
        if rows > 5:
            rows = 5
            cols = 2
        fig, ax, axt, fig_par = FigCreate(
            rows=rows, cols=cols, plot_type=0, paper_col=paper_col
        )
        for i, exp in enumerate(exps):
            for p in range(n_peaks):
                ax[p].plot(
                    exp.dcv_x,
                    exp.dcv_peaks[p, :] * ps,
                    color=clrs[i],
                    linestyle=lnstls[i],
                    label=labels[i],
                )
                ax[p].fill_between(
                    exp.dcv_x,
                    exp.dcv_peaks[p, :] * ps - exp.dcv_peaks_std[p, :],
                    exp.dcv_peaks[p, :] * ps + exp.dcv_peaks_std[p, :],
                    color=clrs[i],
                    alpha=0.3,
                )
        FigSave(
            fig_name + "_dcnv",
            out_path,
            fig,
            ax,
            axt,
            fig_par,
            subfolder=subfolder,
            xLim=xLim,
            yLim=yLim,
            xLab="T [°C]",
            yLab='DTG [% min$^{-1}$]',
            annotate_lttrs=True,
            legend=legend_loc,
            pdf=pdf,
            svg=svg,
        )


def bar_plot(out_path, df, vars_bar = ['Ti_C', 'Tp_C', 'Tb_C'],
             vars_scat = ['S_comb'],
             smpl_labs = None, var_bar_labs=None, var_scat_labs=None,
             yLab='T [°C]', ytLab='S (comb. index)',
             fig_name='Plot', dx_leg=0, dy_leg=0, leg_col = 1,
             xLab=None,  xLim=None,
             yLim=None, ytLim=None, xTicks=None,
             yTicks=None, ytTicks=None, x_lab_rot=30, legend_inside=True,
             leg_loc='best',
             subfolder='', paper_col=1.5, hgt_mltp=1, grid=False):
    htchs = (None, '//', '...', '--', 'O', '\\\\', 'oo', '\\\\\\', '/////', '.....',
             '//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**',
             None, '////', '...', 'o', 'O', '.')

    n_vars_bar = len(vars_bar)
    df_bar = df[vars_bar]
    n_smpls_bar = int(len(df_bar.index)/2)
    df_bar_ave = df_bar.iloc[0:n_smpls_bar, :].copy()
    df_bar_std = df_bar.iloc[n_smpls_bar:, :].copy()

    try:
        df_scat = df[vars_scat].to_frame()
    except AttributeError:
        df_scat = df[vars_scat].copy()
    df_scat_ave = df_scat.iloc[0:int(len(df_scat.index)/2)].copy()
    df_scat_std = df_scat.iloc[int(len(df_scat.index)/2):].copy()

    for ddff in [df_bar_ave, df_bar_std, df_scat_ave, df_scat_std]:
        if smpl_labs:
            ddff.index = smpl_labs
        else:
            ddff.index = [i[:-4] for i in ddff.index]
    if var_bar_labs:
        df_bar_ave.rename(columns=dict(zip(df_bar_ave.columns,var_bar_labs)),
                          inplace=True)
        df_bar_std.rename(columns=dict(zip(df_bar_std.columns,var_bar_labs)),
                          inplace=True)
    if var_scat_labs:
        df_scat_ave.rename(columns=dict(zip(df_scat_ave.columns,var_scat_labs)),
                           inplace=True)
        df_scat_std.rename(columns=dict(zip(df_scat_std.columns,var_scat_labs)),
                           inplace=True)



    fig, ax, axt, fig_par = FigCreate(1, 1, 1, paper_col=paper_col,
                                      hgt_mltp=hgt_mltp)
    if n_vars_bar ==1:
        width = .4
    else:
        width = .9
    df_bar_ave.plot(kind='bar', ax=ax[0], yerr=df_bar_std, capsize=2,
                    ecolor='k', edgecolor='black', width=width)
    bars = ax[0].patches
    patterns = htchs[:n_vars_bar]  # set hatch patterns in the correct order
    hatches = []  # list for hatches in the order of the bars
    for h in patterns:  # loop over patterns to create bar-ordered hatches
        for i in range(int(len(bars) / len(patterns))):
            hatches.append(h)
    for bar, hatch in zip(bars, hatches):  # loop over bars and hatches to set hatches in correct order
        bar.set_hatch(hatch)
    for v, vr in enumerate(df_scat_ave.columns):
        axt[0].scatter(x=df_scat_ave.index,y= df_scat_ave[vr],
                       linestyle='None', marker=mrkrs[0],
                       facecolor=clrs[n_vars_bar + v],
                       edgecolor='k', label=vr, s=70)
        if not all(df_scat_std[vr].isnull()):
            axt[0].errorbar(x=df_scat_ave.index,y=df_scat_ave[vr],
                            yerr=df_scat_std[vr], capsize=2, ls='none',
                            color='black',
                            )


    hnd_ax, lab_ax = ax[0].get_legend_handles_labels()
    hnd_axt, lab_axt = axt[0].get_legend_handles_labels()
    if legend_inside:
        ax[0].legend(hnd_ax + hnd_axt, lab_ax + lab_axt, loc=leg_loc,
                     ncol=leg_col)
    else:

        ax[0].legend(hnd_ax + hnd_axt, lab_ax + lab_axt, loc='center',
                 ncol=leg_col,
                 bbox_to_anchor=(1.2 + dx_leg, .85 + dy_leg))
    if smpl_labs:
        x_lbls = smpl_labs
    else:
        x_lbls = df_bar_ave.index.tolist()
    if x_lab_rot == 0:
        ax[0].set_xticklabels([str(i) for i in x_lbls])
    else:
        ax[0].set_xticklabels([str(i) for i in x_lbls],
                            rotation=x_lab_rot, ha="right",
                            rotation_mode="anchor")
    FigSave(fig_name, out_path, fig, ax, axt, fig_par, tight_layout=True,
            legend=None, subfolder=subfolder,
            xLab=xLab, yLab=yLab, ytLab=ytLab,
            xLim=xLim, yLim=yLim, ytLim=ytLim, xTicks=xTicks,
            yTicks=yTicks, ytTicks=ytTicks, grid=grid
            )




def PathsCreate(subfolder=''):
    ''' This function creates 2 folder paths (independently of the os). \
        For each path checks the folder existence, if False it creates it.
    in_path : pathlib object
        path to the Input folder.
    out_path : pathlib object
        path to the Output folder.'''
    try:  # start from script folder
        script_path = plib.Path(__file__).parents[0]  # script folder path
    except NameError:  # if a cell is run alone and the __file__ is not available
        script_path = plib.Path.cwd()  # the current working directory
    # create all necessary paths to subfolders starting from the script one
    if subfolder == '_test':  # only for the _test work in the _test folder
        in_path = plib.Path(script_path, "_test")
        out_path = plib.Path(script_path, "_test", 'Output')
    else:  # in all other cases
        in_path = plib.Path(script_path, "Input", subfolder)
        out_path = plib.Path(script_path, "Input", subfolder, 'Output')
    # check existence of each folders and create them if missing
    plib.Path(in_path).mkdir(parents=True, exist_ok=True)
    plib.Path(out_path).mkdir(parents=True, exist_ok=True)

    return in_path, out_path  # returns the two object-paths



class tga_exp:

    def __init__(self, name, filenames, folder, label=None, t_moist=38,
                 t_VM=147, Tlims_dtg_C=[120, 800], T_unit='Celsius',
                 correct_ash_mg=None, correct_ash_fr=None,
                 T_intial_C=40, resolution_T_dtg=5, dtg_basis='temperature',
                 dtg_w_SavFil=101, plot_font='Dejavu Sans'):
        self.name = name
        self.filenames = filenames
        if not label:
            self.label = name
        else:
            self.label = label
        self.n_repl = len(filenames)
        self.dataframes = []
        self.processed_data = {}  # Dictionary to store processed data
        self.folder = folder
        self.in_path, self.out_path = PathsCreate(folder)
        self.column_name_mapping = \
            {'Time': 't_min', 'Temperature': 'T_C', 'Weight': 'm_p',
             'Weight.1': 'm_mg', 'Heat Flow': 'heatflow_mW',
             '##Temp./>C': 'T_C', 'Time/min': 't_min', 'Mass/%': 'm_p',
             'Segment': 'segment'}
        self.t_moist = t_moist
        self.t_VM = t_VM
        self.correct_ash_mg = correct_ash_mg
        self.correct_ash_fr = correct_ash_fr
        self.T_unit = T_unit
        self.T_intial_C = T_intial_C
        if self.T_unit == 'Celsius':
            self.T_symbol = '°C'
            self.Tlims_dtg = Tlims_dtg_C
        elif self.T_unit == 'Kelvin':
            self.T_symbol = 'K'
            self.Tlims_dtg = [T + 273.15 for T in Tlims_dtg_C]
        self.resolution_T_dtg = resolution_T_dtg
        self.dtg_basis = dtg_basis
        self.dtg_w_SavFil = dtg_w_SavFil

        self.data_loaded = False  # Flag to track if data is loaded
        self.proximate_computed = False
        self.dtg_computed = False
        self.oxidation_properties_computed = False
        self.solid_dist_computed = False
        # plotting parameters
        self.plot_font = plot_font

    def load_single_file(self, filename):
        path = plib.Path(self.in_path, filename + '.txt')
        if not path.is_file():
            path = plib.Path(self.in_path, filename + '.csv')
        file = pd.read_csv(path, sep='\t')
        if file.shape[1] < 3:
            file = pd.read_csv(path, sep=',')

        file = file.rename(columns={col: self.column_name_mapping.get(col, col)
                                    for col in file.columns})
        for column in file.columns:
            file[column] = pd.to_numeric(file[column], errors='coerce')
        file.dropna(inplace=True)
        self.files = [file]
        self.data_loaded = True  # Flag to track if data is loaded
        return self.files

    def load_replicate_files(self):
        print('\n' + self.name)
        # import files and makes sure that replicates have the same size
        files, len_files,  = [], []
        for i, filename in enumerate(self.filenames):
            print(filename)
            file = self.load_single_file(filename)[0]
            # FILE CORRECTION
            if self.correct_ash_mg:
                file['m_mg'] = file['m_mg'] - np.min(file['m_mg']
                                                     ) + self.correct_ash_mg
            if file['m_mg'].iloc[-1] < 0:
                print('neg. mass correction: Max [mg]',
                      np.round(np.max(file['m_mg']), 3), '; Min [mg]',
                      np.round(np.min(file['m_mg']), 3))
                file['m_mg'] = file['m_mg'] - np.min(file['m_mg'])

            file['m_p'] = file['m_mg']/np.max(file['m_mg'])*100
            if self.correct_ash_fr:
                file['m_p'] = file['m_p'] - np.min(file['m_p']
                                                   ) + self.correct_ash_fr
                file['m_p'] = file['m_p']/np.max(file['m_p'])*100
            file = file[file['T_C'] >= self.T_intial_C].copy()
            file['T_K'] = file['T_C'] + 273.15
            files.append(file)
            len_files.append(max(file.shape))
        self.len_sample = np.min(len_files)
        # keep the shortest vector size for all replicates, create the object
        self.files = [f.head(self.len_sample) for f in files]
        self.data_loaded = True  # Flag to track if data is loaded
        return self.files

    def proximate_analysis(self):
        if not self.data_loaded:
            self.load_replicate_files()

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
            if self.T_unit == 'Celsius':
                self.T_stk[:, f] = file['T_C']
            elif self.T_unit == 'Kelvin':
                self.T_stk[:, f] = file['T_K']
            self.time_stk[:, f] = file['t_min']

            self.m_ar_stk[:, f] = file['m_mg']
            self.mp_ar_stk[:, f] = file['m_p']

            self.idx_moist_stk[f] = np.argmax(self.time_stk[:, f]
                                              > self.t_moist+0.01)
            self.idx_vm_stk[f] = np.argmax(self.time_stk[:, f] > self.t_VM)

            self.moist_ar_stk[f] = 100 - self.mp_ar_stk[self.idx_moist_stk[f],
                                                        f]

            self.ash_ar_stk[f] = self.mp_ar_stk[-1, f]

            self.fc_ar_stk[f] = (self.mp_ar_stk[self.idx_vm_stk[f], f]
                                 - self.ash_ar_stk[f])
            self.vm_ar_stk[f] = (100 - self.moist_ar_stk[f] -
                                 self.ash_ar_stk[f] - self.fc_ar_stk[f])

            self.mp_db_stk[:, f] = self.mp_ar_stk[:, f] * \
                100/(100-self.moist_ar_stk[f])
            self.mp_db_stk[:, f] = np.where(self.mp_db_stk[:, f] > 100, 100,
                                            self.mp_db_stk[:, f])
            self.ash_db_stk[f] = self.ash_ar_stk[f]*100 / \
                (100-self.moist_ar_stk[f])
            self.vm_db_stk[f] = self.vm_ar_stk[f]*100 / (100 -
                                                         self.moist_ar_stk[f])
            self.fc_db_stk[f] = self.fc_ar_stk[f]*100 / (100 -
                                                         self.moist_ar_stk[f])

            self.mp_daf_stk[:, f] = (self.mp_db_stk[:, f] - self.ash_db_stk[f]
                                     )*100/(100-self.ash_db_stk[f])
            self.vm_daf_stk[f] = (self.vm_db_stk[f] - self.ash_db_stk[f]
                                  )*100/(100-self.ash_db_stk[f])
            self.fc_daf_stk[f] = (self.fc_db_stk[f] - self.ash_db_stk[f]
                                  )*100/(100-self.ash_db_stk[f])
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

        self.proximate_computed = True
        return

    def dtg_analysis(self):
        if not self.data_loaded:
            self.load_replicate_files()
        if not self.proximate_computed:
            self.proximate_analysis()
        len_dtg_db = int((self.Tlims_dtg[1] - self.Tlims_dtg[0]
                          )*self.resolution_T_dtg)
        self.T_dtg = np.linspace(self.Tlims_dtg[0], self.Tlims_dtg[1],
                                 len_dtg_db)
        self.time_dtg_stk = np.ones((len_dtg_db, self.n_repl))
        self.mp_db_dtg_stk = np.ones((len_dtg_db, self.n_repl))
        self.dtg_db_stk = np.ones((len_dtg_db, self.n_repl))
        for f, file in enumerate(self.files):
            idxs_dtg = [np.argmax(self.T_stk[:, f] > self.Tlims_dtg[0]),
                        np.argmax(self.T_stk[:, f] > self.Tlims_dtg[1])]
            # T_dtg is taken fixed
            self.T_dtg = np.linspace(self.Tlims_dtg[0], self.Tlims_dtg[1],
                                     len_dtg_db)
            # time start from 0 and consideres a fixed heating rate
            self.time_dtg_stk[:, f] = \
                np.linspace(0, self.time_stk[idxs_dtg[1], f]
                            - self.time_stk[idxs_dtg[0], f], len_dtg_db)

            self.mp_db_dtg_stk[:, f] = \
                np.interp(self.T_dtg, self.T_stk[idxs_dtg[0]: idxs_dtg[1], f],
                          self.mp_db_stk[idxs_dtg[0]: idxs_dtg[1], f])
            # the combusiton indexes use rates as /min
            if self.dtg_basis == 'temperature':
                dtg = np.gradient(self.mp_db_dtg_stk[:, f], self.T_dtg)
            if self.dtg_basis == 'time':
                dtg = np.gradient(self.mp_db_dtg_stk[:, f],
                                  self.time_dtg_stk[:, f])
            self.dtg_db_stk[:, f] = SavFil(dtg, self.dtg_w_SavFil, 1)
        # average
        self.time_dtg = np.average(self.time_dtg_stk, axis=1)
        self.mp_db_dtg = np.average(self.mp_db_dtg_stk, axis=1)
        self.mp_db_dtg_std = np.std(self.mp_db_dtg_stk, axis=1)
        self.dtg_db = np.average(self.dtg_db_stk, axis=1)
        self.dtg_db_std = np.std(self.dtg_db_stk, axis=1)
        self.AveTGstd_p = np.average(self.mp_db_dtg_std)
        self.AveTGstd_p_std = np.nan
        print("Average TG [%] St. Dev. for replicates: " +
              str(round(np.average(self.mp_db_dtg_std), 2)) + " %")
        self.dtg_computed = True
        return self.dtg_db_stk

    def oxidation_properties(self, Tb_thresh=1):
        if not self.data_loaded:
            self.load_replicate_files()
        if not self.proximate_computed:
            self.proximate_analysis()
        if not self.dtg_computed:
            self.dtg_analysis()
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
            # Ti = T at which dtg >1 wt%/min after moisture removal
            self.Ti_idx_stk[f] = int(np.argmax(np.abs(self.dtg_db_stk[:, f]) > 1))
            self.Ti_stk[f] = self.T_dtg[self.Ti_idx_stk[f]]
            # Tp is the T of max abs(dtg)
            self.Tp_idx_stk[f] = int(np.argmax(np.abs(self.dtg_db_stk[:, f])))
            self.Tp_stk[f] = self.T_dtg[self.Tp_idx_stk[f]]
            # Tb reaches < 1 wt%/min at end of curve
            try:
                self.Tb_idx_stk[f] = int(np.flatnonzero(self.dtg_db_stk[:, f]
                                                       < -Tb_thresh)[-1])
            except IndexError:  # the curve nevers goes above 1%
                self.Tb_idx_stk[f] = 0
            self.Tb_stk[f] = self.T_dtg[self.Tb_idx_stk[f]]

            self.dwdT_max_stk[f] = np.max(np.abs(self.dtg_db_stk[:, f]))
            self.dwdT_mean_stk[f] = np.average(np.abs(self.dtg_db_stk[:, f]))
            # combustion index
            self.S_stk[f] = (self.dwdT_max_stk[f]*self.dwdT_mean_stk[f] /
                             self.Ti_stk[f]/self.Ti_stk[f]/self.Tb_stk[f])
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
        self.oxidation_properties_computed = True

    def solid_dist(self, steps=[40, 70, 100, 130, 160, 190],
                      print_dfs=True):
        """
        produces tg and dtg plots of each replicate in a sample. Quality of plots
        is supposed to allow checking for errors.

        """
        if not self.data_loaded:
            self.load_replicate_files()
        if not self.proximate_computed:
            self.proximate_analysis()
        if not self.dtg_computed:
            self.dtg_analysis()
        plib.Path(self.out_path, 'SolidDist').mkdir(parents=True,
                                                       exist_ok=True)
        out_path_MS = plib.Path(self.out_path, 'SolidDist')
        self.dist_steps = steps + ['end']
        len_dist_step = len(self.dist_steps)

        self.idxs_dist_steps_stk = np.ones((len_dist_step, self.n_repl))
        self.T_dist_stk = np.ones((len_dist_step, self.n_repl))
        self.time_dist_stk = np.ones((len_dist_step, self.n_repl))
        self.dmp_dist_stk = np.ones((len_dist_step, self.n_repl))
        self.loc_dist_stk = np.ones((len_dist_step, self.n_repl))

        for f, file in enumerate(self.files):
            idxs = []
            for step in steps:
                idxs.append(np.argmax(self.time > step))
            self.idxs_dist_steps_stk[:, f] = idxs.append(len(self.time)-1)
            self.T_dist_stk[:, f] = self.T_stk[idxs, f]
            self.time_dist_stk[:, f] = self.time_stk[idxs, f]

            self.dmp_dist_stk[:, f] = -np.diff(self.mp_db[idxs], prepend=100)

            self.loc_dist_stk[:, f] = \
                np.convolve(np.insert(self.mp_db[idxs], 0, 100),
                            [.5, .5], mode='valid')
        self.T_dist = np.average(self.T_dist_stk, axis=1)
        self.T_dist_std = np.std(self.T_dist_stk, axis=1)
        self.time_dist = np.average(self.time_dist_stk, axis=1)
        self.time_dist_std = np.std(self.time_dist_stk, axis=1)
        self.dmp_dist = np.average(self.dmp_dist_stk, axis=1)
        self.dmp_dist_std = np.std(self.dmp_dist_stk, axis=1)
        self.loc_dist = np.average(self.loc_dist_stk, axis=1)
        self.loc_dist_std = np.std(self.loc_dist_stk, axis=1)
        self.solid_dist_computed =True

    def report_solid_dist(self):
        if not self.data_loaded:
            self.load_replicate_files()
        if not self.proximate_computed:
            self.proximate_analysis()
        if not self.dtg_computed:
            self.dtg_analysis()
        if not self.oxidation_properties_computed:
            self.oxidation_properties()
        if not self.solid_dist_computed:
            self.solid_dist()
        out_path_rep = plib.Path(self.out_path, 'SingleReports')
        out_path_rep.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(index=self.dist_steps)
        df['t_ave'] = self.T_dist
        df['t_std'] = self.T_dist_std
        df['time_ave'] = self.time_dist
        df['time_std'] = self.time_dist_std
        df['dmp_ave'] = self.dmp_dist
        df['dmp_std'] = self.dmp_dist_std
        df['loc_ave'] = self.loc_dist
        df['loc_std'] = self.loc_dist_std
        self.report_dist = df
        df.to_excel(plib.Path(out_path_rep, self.name + '_SolidDist.xlsx'))
        return self.report_dist

    def report(self):
        if not self.data_loaded:
            self.load_replicate_files()
        if not self.proximate_computed:
            self.proximate_analysis()
        if not self.dtg_computed:
            self.dtg_analysis()
        if not self.oxidation_properties_computed:
            self.oxidation_properties()
        out_path_rep = plib.Path(self.out_path, 'SingleReports')
        out_path_rep.mkdir(parents=True, exist_ok=True)
        if self.T_unit == 'Celsius':
            TiTpTb = ['Ti_C', 'Tp_C', 'Tb_C']
        elif self.T_unit == 'Kelvin':
            TiTpTb = ['Ti_K', 'Tp_K', 'Tb_K']

        columns = ['moist_ar_p', 'ash_ar_p', 'ash_db_p', 'vm_db_p', 'fc_db_p',
                   'vm_daf_p', 'fc_daf_p'] + TiTpTb + \
            ['idx_dwdT_max_p_min', 'dwdT_mean_p_min', 'S_comb', 'AveTGstd_p']
        report = pd.DataFrame(index=self.filenames, columns=columns)

        for f, filename in enumerate(self.filenames):
            report.loc[filename] = [
                self.moist_ar_stk[f], self.ash_ar_stk[f], self.ash_db_stk[f],
                self.vm_db_stk[f], self.fc_db_stk[f], self.vm_daf_stk[f],
                self.fc_daf_stk[f], self.Ti_stk[f], self.Tp_stk[f], self.Tb_stk[f],
                self.dwdT_max_stk[f], self.dwdT_mean_stk[f], self.S_stk[f], np.nan
            ]

        report.loc['average'] = [
            self.moist_ar, self.ash_ar, self.ash_db, self.vm_db, self.fc_db,
            self.vm_daf, self.fc_daf, self.Ti, self.Tp, self.Tb, self.dwdT_max,
            self.dwdT_mean, self.S, self.AveTGstd_p
        ]

        report.loc['std'] = [
            self.moist_ar_std, self.ash_ar_std, self.ash_db_std, self.vm_db_std,
            self.fc_db_std, self.vm_daf_std, self.fc_daf_std, self.Ti_std,
            self.Tp_std, self.Tb_std, self.dwdT_max_std, self.dwdT_mean_std,
            self.S_std, self.AveTGstd_p_std
        ]

        self.report = report

        report.to_excel(plib.Path(out_path_rep, self.name + '_ProxOxid.xlsx'))
        return report

    def plt_sample_tg(self, TG_lab='TG [wt%]', grid=False):
        """
        produces tg and dtg plots of each replicate in a sample. Quality of plots
        is supposed to allow checking for errors.

        """
        out_path_ST = plib.Path(self.out_path, 'SingleSamples')
        out_path_ST.mkdir(parents=True, exist_ok=True)
        fig_name = self.name
        fig, ax, axt, fig_par = FigCreate(rows=3, cols=1, plot_type=0,
                                          paper_col=1, font=self.plot_font)
        for f in range(self.n_repl):
            ax[0].plot(self.time_stk[:, f], self.T_stk[:, f], color=clrs[f],
                       linestyle=lnstls[f], label=self.filenames[f])
            ax[1].plot(self.time_stk[:, f], self.mp_ar_stk[:, f], color=clrs[f],
                       linestyle=lnstls[f])
            ax[2].plot(self.time_stk[:, f], self.mp_db_stk[:, f], color=clrs[f],
                       linestyle=lnstls[f])
            ax[0].vlines(self.time_stk[self.idx_moist_stk[f], f],
                         self.T_stk[self.idx_moist_stk[f], f] - 50,
                         self.T_stk[self.idx_moist_stk[f], f] + 50,
                         linestyle=lnstls[f], color=clrs[f])
            ax[1].vlines(self.time_stk[self.idx_moist_stk[f], f],
                          self.mp_ar_stk[self.idx_moist_stk[f], f] - 5,
                          self.mp_ar_stk[self.idx_moist_stk[f], f] + 5,
                          linestyle=lnstls[f], color=clrs[f])
            if self.vm_db < 99:
                ax[0].vlines(self.time_stk[self.idx_vm_stk[f], f],
                             self.T_stk[self.idx_vm_stk[f], f] - 50,
                             self.T_stk[self.idx_vm_stk[f], f] + 50,
                             linestyle=lnstls[f], color=clrs[f])
                ax[2].vlines(self.time_stk[self.idx_vm_stk[f], f],
                             self.mp_db_stk[self.idx_vm_stk[f], f] - 5,
                             self.mp_db_stk[self.idx_vm_stk[f], f] + 5,
                             linestyle=lnstls[f], color=clrs[f])
        ax[0].legend(loc='best')
        FigSave(fig_name + '_TG', out_path_ST, fig, ax, axt, fig_par,
                xLab='time [min]',
                yLab=['T ['+self.T_symbol+']',
                      TG_lab+'(stb)', TG_lab+'(db)'], grid=grid)


    def plt_sample_dtg(self,
                       TG_lab='TG [wt%]', DTG_lab=None, grid=False):
        """
        produces tg and dtg plots of each replicate in a sample. Quality of
        plots is supposed to allow checking for errors.

        """
        if DTG_lab is None:
            if self.dtg_basis == 'temperature':
                DTG_lab = 'DTG [wt%/' + self.T_symbol + ']'
            elif self.dtg_basis == 'time':
                DTG_lab='DTG [wt%/min]'
        out_path_ST = plib.Path(self.out_path, 'SingleSamples')
        out_path_ST.mkdir(parents=True, exist_ok=True)
        fig_name = self.name

        fig, ax, axt, fig_par = FigCreate(rows=3, cols=1, plot_type=0,
                                          paper_col=1, font=self.plot_font)
        for f in range(self.n_repl):
            ax[0].plot(self.time_dtg, self.T_dtg, color=clrs[f],
                       linestyle=lnstls[f], label=self.filenames[f])
            ax[1].plot(self.time_dtg, self.mp_db_dtg_stk[:, f], color=clrs[f],
                       linestyle=lnstls[f])
            ax[2].plot(self.time_dtg, self.dtg_db_stk[:, f], color=clrs[f],
                       linestyle=lnstls[f])
            if self.oxidation_properties_computed:
                ax[2].vlines(self.time_dtg[self.Ti_idx_stk[f]],
                             ymin=-1.5, ymax=0,
                             linestyle=lnstls[f], color=clrs[f], label='Ti')
                ax[2].vlines(self.time_dtg[self.Tp_idx_stk[f]],
                             ymin=np.min(self.dtg_db_stk[:, f]),
                             ymax=np.min(self.dtg_db_stk[:, f])/5,
                             linestyle=lnstls[f], color=clrs[f], label='Tp')
                ax[2].vlines(self.time_dtg[self.Tb_idx_stk[f]],
                             ymin=-1.5, ymax=0,
                             linestyle=lnstls[f], color=clrs[f], label='Tb')
        ax[0].legend(loc='best')
        FigSave(fig_name + '_DTGstk', out_path_ST, fig, ax, axt, fig_par,
                xLab='time [min]',
                yLab=['T ['+self.T_symbol+']', TG_lab + '(db)',
                      DTG_lab + '(db)'], grid=grid)

    def plt_solid_dist(self, paper_col=1, hgt_mltp=1.25, TG_lab='TG [wt%]',
                       grid=False):

        out_path_ST = plib.Path(self.out_path, 'SingleSamples')
        out_path_ST.mkdir(parents=True, exist_ok=True)
        fig_name = self.name
        fig, ax, axt, fig_par = FigCreate(rows=2, cols=1, plot_type=0,
                                          paper_col=paper_col,
                                          hgt_mltp=hgt_mltp)

        ax[0].plot(self.time, self.T)
        # ax[0].plot(self.time, np.diff(self.T, prepend=np.nan)*100, color=clrs[i],
        #            linestyle=lnstls[i], label=labels[i])
        ax[0].fill_between(self.time, self.T - self.T_std,
                           self.T + self.T_std,
                           alpha=.3)
        ax[1].plot(self.time, self.mp_db)
        ax[1].fill_between(self.time, self.mp_db - self.mp_db_std,
                           self.mp_db + self.mp_db_std,
                           alpha=.3)
        for tm, mp, dmp in zip(self.time_dist, self.loc_dist,
                               self.dmp_dist):
            ax[1].annotate(str(np.round(dmp, 0)) + '%',
                           ha='center', va='top',
                           xy=(tm - 10, mp+1), fontsize=9)
        FigSave(fig_name + '_dist', out_path_ST, fig, ax, axt, fig_par,
                xLab='time [min]',
                yLab=['T ['+self.T_symbol+']', TG_lab+'(db)'],
                grid=grid
                )

def plt_tgs(exps, fig_name='Fig', paper_col=.78, hgt_mltp=1.25,
            xLim=None, yLim_tg=[0, 100], yTicks_tg=None, grid=False,
            TG_lab='TG [wt%]', lttrs=False, pdf=False, svg=False):
    out_path_TGs = plib.Path(exps[0].out_path, 'TGs')
    out_path_TGs.mkdir(parents=True, exist_ok=True)
    fig, ax, axt, fig_par = FigCreate(rows=1, cols=1, plot_type=0,
                                      paper_col=paper_col,
                                      hgt_mltp=hgt_mltp)
    for i, exp in enumerate(exps):
        ax[0].plot(exp.T, exp.mp_db, color=clrs[i], linestyle=lnstls[i],
                   label=exp.label if exp.label else exp.name)
        ax[0].fill_between(exp.T, exp.mp_db - exp.mp_db_std,
                           exp.mp_db + exp.mp_db_std, color=clrs[i],
                           alpha=.3)
    FigSave(fig_name + '_tg', out_path_TGs, fig, ax, axt, fig_par,
            xLim=xLim, yLim=yLim_tg,
            yTicks=yTicks_tg,
            xLab='T [' + exps[0].T_symbol + ']', legend='upper right',
            yLab=TG_lab, annotate_lttrs=lttrs, grid=grid, pdf=pdf, svg=svg)


def plt_dtgs(exps, fig_name='Fig', paper_col=.78, hgt_mltp=1.25,
             xLim=None, yLim_dtg=None, yTicks_dtg=None, grid=False,
             DTG_lab=None, lttrs=False, plt_gc=False, gc_Tlim=300,
             pdf=False, svg=False):

    out_path_DTGs = plib.Path(exps[0].out_path, 'DTGs')
    out_path_DTGs.mkdir(parents=True, exist_ok=True)
    if not DTG_lab:
        DTG_lab = 'DTG [wt%/' + exps[0].T_symbol + ']'

    fig, ax, axt, fig_par = FigCreate(rows=1, cols=1, plot_type=0,
                                      paper_col=paper_col,
                                      hgt_mltp=hgt_mltp)
    for i, exp in enumerate(exps):
        ax[0].plot(exp.T_dtg, exp.dtg_db, color=clrs[i], linestyle=lnstls[i],
                   label=exp.label if exp.label else exp.name)
        ax[0].fill_between(exp.T_dtg, exp.dtg_db - exp.dtg_db_std,
                           exp.dtg_db + exp.dtg_db_std, color=clrs[i],
                           alpha=.3)
    if plt_gc:
        ax[0].vlines(gc_Tlim, ymin=yLim_dtg[0], ymax=yLim_dtg[1],
                     linestyle=lnstls[1], color=clrs[7],
                     label='T$_{max GC-MS}$')
    ax[0].legend(loc='lower right')
    FigSave(fig_name + '_dtg', out_path_DTGs, fig, ax, axt, fig_par,
            xLim=xLim, yLim=yLim_dtg,
            yTicks=yTicks_dtg,
            yLab=DTG_lab, xLab='T [' + exps[0].T_symbol + ']',
            pdf=pdf, svg=svg, annotate_lttrs=lttrs, grid=grid)


def plt_cscds(exps, fig_name='Fig', paper_col=.78, hgt_mltp=1.25,
              xLim=None,clrs_cscd=False,  yLim0_cscd=-8,
              shifts_cscd=np.asarray([0, 11, 5, 10, 10, 10, 11]),
              peaks_cscd=None, peak_names=None, dh_names_cscd=0.1,
              loc_names_cscd=130,
              hgt_mltp_cscd=1.5, legend_cscd='lower right',
              y_values_cscd=[-10, 0],
              lttrs=False, DTG_lab=None, pdf=False, svg=False):
    out_path_CSCDs = plib.Path(exps[0].out_path, 'DTGs')
    out_path_CSCDs.mkdir(parents=True, exist_ok=True)
    labels = [exp.label if exp.label else exp.name for exp in exps]
    if not DTG_lab:
        DTG_lab = 'DTG [wt%/' + exps[0].T_symbol + ']'
    yLim_cscd = [yLim0_cscd, np.sum(shifts_cscd)]
    dh = np.cumsum(shifts_cscd)
    fig, ax, axt, fig_par = FigCreate(1, 1, paper_col=.78,
                                      hgt_mltp=hgt_mltp_cscd)
    for n, exp in enumerate(exps):
        if clrs_cscd:
            ax[0].plot(exp.T_dtg, exp.dtg_db + dh[n], color=clrs[n],
                       linestyle=lnstls[0])
            ax[0].fill_between(exp.T_fit, exp.dtg_db - exp.dtg_db_std
                               + dh[n], exp.dtg_db + exp.dtg_db_std + dh[n],
                               color=clrs[n], alpha=.3)
        else:
            ax[0].plot(exp.T_dtg, exp.dtg_db + dh[n], color='k',
                       linestyle=lnstls[0])
            ax[0].fill_between(exp.T_dtg, exp.dtg_db - exp.dtg_db_std
                               + dh[n], exp.dtg_db + exp.dtg_db_std + dh[n],
                               color='k', alpha=.3)
        ax[0].annotate(labels[n], ha='left', va='bottom',
                       xy=(loc_names_cscd,
                           exp.dtg_db[np.argmax(exp.T_dtg>
                                                 loc_names_cscd)] +
                                      dh[n] + dh_names_cscd))
    if peaks_cscd:
        for p, peak in enumerate(peaks_cscd):
            if peak:  # to allow to use same markers by skipping peaks
                ax[0].plot(peak[0], peak[1], linestyle='None',
                           marker=mrkrs[p], color='k', label=peak_names[p])
    if y_values_cscd:
        ax[0].set_yticks(y_values_cscd)
    else:
        ax[0].set_yticks([])
    FigSave(fig_name + '_cscd', out_path_CSCDs, fig, ax, axt, fig_par,
            legend=legend_cscd, annotate_lttrs=lttrs,
            xLab='T [' + exps[0].T_symbol + ']', yLab=DTG_lab,
            xLim=xLim, yLim=yLim_cscd, svg=svg, pdf=pdf)


def print_reports(exps, filename='Rep'):
    out_path_REPs = plib.Path(exps[0].out_path, 'DTGs')
    out_path_REPs.mkdir(parents=True, exist_ok=True)
    reports = pd.DataFrame(columns=list(exps[0].report))
    for i, exp in enumerate(exps):
        reports.loc[exp.label + '_ave'] = exp.report.loc['average', :]
    for i, exp in enumerate(exps):
        reports.loc[exp.label + '_std'] = exp.report.loc['std', :]
    reports.to_excel(plib.Path(out_path_REPs, filename + '.xlsx'))
    return reports


def plt_solid_dists(exps, fig_name="Dist",
                  TG_lab='TG [wt%]', DTG_lab='DTG [wt%/min]',
                  hgt_mltp=1.25, paper_col=.78, labels=None,
                  xLim=None, yLim=[[0, 1000], [0, 100]], yTicks=None, lttrs=False,
                  print_dfs=True, grid=False,):
    """
    produces tg and dtg plots of each replicate in a sample. Quality of plots
    is supposed to allow checking for errors.

    """
    plib.Path(exps[0].out_path, 'SolidDist').mkdir(parents=True, exist_ok=True)
    out_path_MS = plib.Path(exps[0].out_path, 'SolidDist')
    if not labels:  # try with labels and use name if no label is given
        labels = [exp.label if exp.label else exp.name for exp in exps]
    fig, ax, axt, fig_par = FigCreate(rows=2, cols=1, plot_type=0,
                                      paper_col=paper_col,
                                      hgt_mltp=hgt_mltp)
    for i, exp in enumerate(exps):


        ax[0].plot(exp.time, exp.T, color=clrs[i],
                   linestyle=lnstls[i], label=labels[i])
        ax[0].fill_between(exp.time, exp.T - exp.T_std,
                           exp.T + exp.T_std, color=clrs[i],
                           alpha=.3)
        ax[1].plot(exp.time, exp.mp_db, color=clrs[i],
                   linestyle=lnstls[i], label=labels[i])
        ax[1].fill_between(exp.time, exp.mp_db - exp.mp_db_std,
                           exp.mp_db + exp.mp_db_std, color=clrs[i],
                           alpha=.3)
        for tm, mp, dmp in zip(exp.time_dist, exp.loc_dist,
                               exp.dmp_dist):
            ax[1].annotate(str(np.round(dmp, 0)) + '%',
                           ha='center', va='top',
                           xy=(tm - 10, mp+1), fontsize=9, color=clrs[i])
        ax[0].legend(loc='upper left')
        ax[1].legend(loc='center left')
    FigSave(fig_name + 'dist_db', out_path_MS, fig, ax, axt, fig_par,
            xLim=xLim, yLim=yLim, yTicks=yTicks, xLab='time [min]',
            yLab=['T [' + exps[0].T_symbol + ']', TG_lab+'(db)'],
            annotate_lttrs=lttrs, grid=grid)

# Main script
if __name__ == "__main__":
    folder = '_test'
    CLS = tga_exp(folder=folder, name='CLS',
                  filenames=['CLSOx5_1', 'CLSOx5_2', 'CLSOx5_3'],
                  t_moist=38, t_VM=147, T_unit='Celsius')
    e = CLS.report()

    CLS.plt_sample_tg()
    CLS.plt_sample_dtg()
    MIS = tga_exp(folder=folder, name='MIS',
                  filenames=['MIS_1', 'MIS_2', 'MIS_3'],
                  t_moist=38, t_VM=147, T_unit='Celsius')
    e = MIS.report()

    MIS.plt_sample_tg()
    MIS.plt_sample_dtg()
    SDa = tga_exp(folder=folder, name='SDa',
                  filenames=['SDa_1', 'SDa_2', 'SDa_3'],
                  t_moist=38, t_VM=147, T_unit='Celsius')
    e = SDa.report()
    f = SDa.solid_dist()
    SDa.plt_solid_dist()
    SDb = tga_exp(folder=folder, name='SDb',
                  filenames=['SDb_1', 'SDb_2', 'SDb_3'],
                  t_moist=38, t_VM=147, T_unit='Celsius')
    e = SDb.report()
    SDb.report_solid_dist()
    SDb.plt_solid_dist()
    # SD.plt_sample_tg()
    # SD.plt_sample_dtg()
#%%

plt_tgs([CLS, MIS, SDa, SDb])
plt_dtgs([CLS, MIS, SDa, SDb])
plt_cscds([CLS, MIS, SDa, SDb])
rep = print_reports([CLS, MIS, SDa, SDb])
plt_solid_dists([SDa, SDb])