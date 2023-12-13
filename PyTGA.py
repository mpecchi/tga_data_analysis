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


def FigSave(filename, out_path, fig, lst_ax, lst_axt, fig_par,
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
    filename : str
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
        plt.savefig(plib.Path(out_path, filename + '.png'), dpi=300,
                    transparent=transparency)
        if pdf is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + '.pdf'))
        if svg is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + '.svg'))
        if eps is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + '.eps'))
    else:  # margins are not given, use a tight layout option and save
        plt.savefig(plib.Path(out_path, filename + '.png'),
                    bbox_inches="tight", dpi=300, transparent=transparency)
        if pdf is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + '.pdf'),
                        bbox_inches="tight")
        if svg is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + '.svg'),
                        bbox_inches="tight")
        if eps is not False:  # save also as pdf
            plt.savefig(plib.Path(out_path, filename + '.eps'),
                        bbox_inches="tight")
    # add the title after saving, so it's only visible in the console
    if title is True:
        lst_ax[0].annotate(filename, xycoords='axes fraction', size='small',
                            xy=(0, 0), xytext=(0.05, .95), clip_on=True)
    return


class tga_exp:

    def __init__(self, name, filenames, folder, load_skiprows=0,
                 label=None, t_moist=38,
                 t_VM=147, Tlims_dtg_C=[120, 800], T_unit='Celsius',
                 correct_ash_mg=None, correct_ash_fr=None,
                 T_intial_C=40, resolution_T_dtg=5, dtg_basis='temperature',
                 dtg_w_SavFil=101, oxid_Tb_thresh=1, plot_font='Dejavu Sans',
                 ):
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
        self.load_skiprows = load_skiprows
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
        self.oxid_Tb_thresh = oxid_Tb_thresh
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
        # plotting parameters
        self.plot_font = plot_font

    def load_single_file(self, filename):
        path = plib.Path(self.in_path, filename + '.txt')
        if not path.is_file():
            path = plib.Path(self.in_path, filename + '.csv')
        file = pd.read_csv(path, sep='\t', skiprows=self.load_skiprows)
        if file.shape[1] < 3:
            file = pd.read_csv(path, sep=',', skiprows=self.load_skiprows)

        file = file.rename(columns={col: self.column_name_mapping.get(col, col)
                                    for col in file.columns})
        for column in file.columns:
            file[column] = pd.to_numeric(file[column], errors='coerce')
        file.dropna(inplace=True)
        self.files = [file]
        self.data_loaded = True  # Flag to track if data is loaded
        return self.files

    def load_files(self):
        print('\n' + self.name)
        # import files and makes sure that replicates have the same size
        files, len_files,  = [], []
        for i, filename in enumerate(self.filenames):
            print(filename)
            file = self.load_single_file(filename)[0]
            # FILE CORRECTION
            if self.correct_ash_mg is not None:
                file['m_mg'] = file['m_mg'] - np.min(file['m_mg']
                                                     ) + self.correct_ash_mg
            try:
                if file['m_mg'].iloc[-1] < 0:
                    print('neg. mass correction: Max [mg]',
                          np.round(np.max(file['m_mg']), 3), '; Min [mg]',
                          np.round(np.min(file['m_mg']), 3))
                    file['m_mg'] = file['m_mg'] - np.min(file['m_mg'])
            except KeyError:
                file['m_mg'] = file['m_p']
            file['m_p'] = file['m_mg']/np.max(file['m_mg'])*100
            if self.correct_ash_fr is not None:
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
            if self.T_unit == 'Celsius':
                self.T_stk[:, f] = file['T_C']
            elif self.T_unit == 'Kelvin':
                self.T_stk[:, f] = file['T_K']
            self.time_stk[:, f] = file['t_min']

            self.m_ar_stk[:, f] = file['m_mg']
            self.mp_ar_stk[:, f] = file['m_p']

            self.idx_moist_stk[f] = np.argmax(self.time_stk[:, f]
                                              > self.t_moist+0.01)


            self.moist_ar_stk[f] = 100 - self.mp_ar_stk[self.idx_moist_stk[f],
                                                        f]
            self.ash_ar_stk[f] = self.mp_ar_stk[-1, f]
            self.mp_db_stk[:, f] = self.mp_ar_stk[:, f] * \
                100/(100-self.moist_ar_stk[f])
            self.mp_db_stk[:, f] = np.where(self.mp_db_stk[:, f] > 100, 100,
                                            self.mp_db_stk[:, f])
            self.ash_db_stk[f] = self.ash_ar_stk[f]*100 / \
                (100-self.moist_ar_stk[f])
            self.mp_daf_stk[:, f] = (self.mp_db_stk[:, f] - self.ash_db_stk[f]
                                     )*100/(100-self.ash_db_stk[f])
            if self.t_VM is not None:
                self.idx_vm_stk[f] = np.argmax(self.time_stk[:, f] > self.t_VM)
                self.fc_ar_stk[f] = (self.mp_ar_stk[self.idx_vm_stk[f], f]
                                    - self.ash_ar_stk[f])
                self.vm_ar_stk[f] = (100 - self.moist_ar_stk[f] -
                                    self.ash_ar_stk[f] - self.fc_ar_stk[f])
                self.vm_db_stk[f] = self.vm_ar_stk[f]*100 / (100 -
                                                            self.moist_ar_stk[f])
                self.fc_db_stk[f] = self.fc_ar_stk[f]*100 / (100 -
                                                            self.moist_ar_stk[f])


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

        self.len_dtg_db = int((self.Tlims_dtg[1] - self.Tlims_dtg[0]
                          )*self.resolution_T_dtg)
        self.T_dtg = np.linspace(self.Tlims_dtg[0], self.Tlims_dtg[1],
                                 self.len_dtg_db)
        self.time_dtg_stk = np.ones((self.len_dtg_db, self.n_repl))
        self.mp_db_dtg_stk = np.ones((self.len_dtg_db, self.n_repl))
        self.dtg_db_stk = np.ones((self.len_dtg_db, self.n_repl))
        for f, file in enumerate(self.files):
            idxs_dtg = [np.argmax(self.T_stk[:, f] > self.Tlims_dtg[0]),
                        np.argmax(self.T_stk[:, f] > self.Tlims_dtg[1])]
            # T_dtg is taken fixed
            self.T_dtg = np.linspace(self.Tlims_dtg[0], self.Tlims_dtg[1],
                                     self.len_dtg_db)
            # time start from 0 and consideres a fixed heating rate
            self.time_dtg_stk[:, f] = \
                np.linspace(0, self.time_stk[idxs_dtg[1], f]
                            - self.time_stk[idxs_dtg[0], f], self.len_dtg_db)

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
        self.proximate_computed = True

    def oxidation_analysis(self):
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
            # Ti = T at which dtg >1 wt%/min after moisture removal
            self.Ti_idx_stk[f] = int(np.argmax(np.abs(self.dtg_db_stk[:, f]) > 1))
            self.Ti_stk[f] = self.T_dtg[self.Ti_idx_stk[f]]
            # Tp is the T of max abs(dtg)
            self.Tp_idx_stk[f] = int(np.argmax(np.abs(self.dtg_db_stk[:, f])))
            self.Tp_stk[f] = self.T_dtg[self.Tp_idx_stk[f]]
            # Tb reaches < 1 wt%/min at end of curve
            try:
                self.Tb_idx_stk[f] = \
                    int(np.flatnonzero(self.dtg_db_stk[:, f] < -
                                       self.oxid_Tb_thresh)[-1])
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
        self.oxidation_computed = True

    def soliddist_analysis(self, steps_min=[40, 70, 100, 130, 160, 190]):
        if not self.proximate_computed:
            self.proximate_analysis()
        self.dist_steps_min = steps_min + ['end']
        len_dist_step = len(self.dist_steps_min)

        self.idxs_dist_steps_min_stk = np.ones((len_dist_step, self.n_repl))
        self.T_dist_stk = np.ones((len_dist_step, self.n_repl))
        self.time_dist_stk = np.ones((len_dist_step, self.n_repl))
        self.dmp_dist_stk = np.ones((len_dist_step, self.n_repl))
        self.loc_dist_stk = np.ones((len_dist_step, self.n_repl))

        for f, file in enumerate(self.files):
            idxs = []
            for step in steps_min:
                idxs.append(np.argmax(self.time_stk[:, f] > step))
            self.idxs_dist_steps_min_stk[:, f] = idxs.append(len(self.time)-1)
            self.T_dist_stk[:, f] = self.T_stk[idxs, f]
            self.time_dist_stk[:, f] = self.time_stk[idxs, f]

            self.dmp_dist_stk[:, f] = -np.diff(self.mp_db_stk[idxs, f],
                                               prepend=100)

            self.loc_dist_stk[:, f] = \
                np.convolve(np.insert(self.mp_db_stk[idxs, f], 0, 100),
                            [.5, .5], mode='valid')
        self.T_dist = np.average(self.T_dist_stk, axis=1)
        self.T_dist_std = np.std(self.T_dist_stk, axis=1)
        self.time_dist = np.average(self.time_dist_stk, axis=1)
        self.time_dist_std = np.std(self.time_dist_stk, axis=1)
        self.dmp_dist = np.average(self.dmp_dist_stk, axis=1)
        self.dmp_dist_std = np.std(self.dmp_dist_stk, axis=1)
        self.loc_dist = np.average(self.loc_dist_stk, axis=1)
        self.loc_dist_std = np.std(self.loc_dist_stk, axis=1)
        self.soliddist_computed = True

    def _prepare_deconvolution_model(self, centers, sigmas, amplitudes, c_mins,
                                     c_maxs, s_mins, s_maxs, a_mins, a_maxs):
        model = LinearModel(prefix="bkg_")
        params = model.make_params(intercept=0, slope=0, vary=False)

        for i in range(len(centers)):
            prefix = f'peak{i}_'
            peak_model = GaussianModel(prefix=prefix)
            pars = peak_model.make_params()
            pars[prefix + 'center'].set(value=centers[i], min=c_mins[i],
                                        max=c_maxs[i])
            pars[prefix + 'sigma'].set(value=sigmas[i], min=s_mins[i],
                                       max=s_maxs[i])
            pars[prefix + 'amplitude'].set(value=amplitudes[i], min=a_mins[i],
                                           max=a_maxs[i])
            model += peak_model
            params.update(pars)

        return model, params

    def deconv_analysis(self, centers, sigmas=None, amplitudes=None,
                        c_mins=None, c_maxs=None, s_mins=None, s_maxs=None,
                        a_mins=None, a_maxs=None, TLim=None):
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
            model, params = \
                self._prepare_deconvolution_model(centers, sigmas, amplitudes,
                                                  c_mins, c_maxs, s_mins,
                                                  s_maxs, a_mins, a_maxs)
            result = model.fit(y, params=params, x=self.T_dtg)
            self.dcv_best_fit_stk[:, f] = -result.best_fit
            self.dcv_r2_stk[f] = 1 - result.residual.var() / np.var(y)
            components = result.eval_components(x=self.T_dtg)
            for p in range(n_peaks):
                prefix = f'peak{p}_'
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
        if not self.proximate_computed:
            self.proximate_analysis()

        out_path = plib.Path(self.out_path, 'SingleSampleReports')
        out_path.mkdir(parents=True, exist_ok=True)

        columns = ['moist_ar_p', 'ash_ar_p', 'ash_db_p', 'vm_db_p', 'fc_db_p',
                   'vm_daf_p', 'fc_daf_p', 'AveTGstd_p']
        rep = pd.DataFrame(index=self.filenames, columns=columns)

        for f, filename in enumerate(self.filenames):
            rep.loc[filename] = [self.moist_ar_stk[f], self.ash_ar_stk[f],
                                 self.ash_db_stk[f], self.vm_db_stk[f],
                                 self.fc_db_stk[f], self.vm_daf_stk[f],
                                 self.fc_daf_stk[f], np.nan]

        rep.loc['ave'] = [self.moist_ar, self.ash_ar, self.ash_db,
                          self.vm_db, self.fc_db, self.vm_daf, self.fc_daf,
                          self.AveTGstd_p]
        rep.loc['std'] = [self.moist_ar_std, self.ash_ar_std, self.ash_db_std,
                          self.vm_db_std, self.fc_db_std, self.vm_daf_std,
                          self.fc_daf_std, self.AveTGstd_p_std]
        self.proximate = rep
        rep.to_excel(plib.Path(out_path, self.name + '_proximate.xlsx'))
        self.proximate_report_computed = True
        return self.proximate

    def oxidation_report(self):
        if not self.oxidation_computed:
            self.oxidation_analysis()
        out_path = plib.Path(self.out_path, 'SingleSampleReports')
        out_path.mkdir(parents=True, exist_ok=True)
        if self.T_unit == 'Celsius':
            TiTpTb = ['Ti_C', 'Tp_C', 'Tb_C']
        elif self.T_unit == 'Kelvin':
            TiTpTb = ['Ti_K', 'Tp_K', 'Tb_K']
        columns = TiTpTb + ['idx_dwdT_max_p_min', 'dwdT_mean_p_min', 'S_comb']
        rep = pd.DataFrame(index=self.filenames, columns=columns)

        for f, filename in enumerate(self.filenames):
            rep.loc[filename] = [self.Ti_stk[f], self.Tp_stk[f],
                                 self.Tb_stk[f], self.dwdT_max_stk[f],
                                 self.dwdT_mean_stk[f], self.S_stk[f]]

        rep.loc['ave'] = [self.Ti, self.Tp, self.Tb, self.dwdT_max,
                          self.dwdT_mean, self.S]

        rep.loc['std'] = [self.Ti_std, self.Tp_std, self.Tb_std,
                          self.dwdT_max_std, self.dwdT_mean_std, self.S_std]
        self.oxidation = rep
        rep.to_excel(plib.Path(out_path, self.name + '_oxidation.xlsx'))
        self.oxidation_report_computed = True
        return self.oxidation

    def soliddist_report(self):
        if not self.soliddist_computed:
            self.soliddist_analysis()
        out_path = plib.Path(self.out_path, 'SingleSampleReports')
        out_path.mkdir(parents=True, exist_ok=True)
        columns = ['T [' + self.T_symbol + '](' + str(s) + 'min)'
                   for s in self.dist_steps_min
                   ] + ['dmp (' + str(s) + 'min)' for s in self.dist_steps_min]
        rep = pd.DataFrame(index=self.filenames, columns=columns)
        for f, filename in enumerate(self.filenames):
            rep.loc[filename] = np.concatenate([self.T_dist_stk[:, f],
                                                self.dmp_dist_stk[:, f]]
                                               ).tolist()
        rep.loc['ave'] = np.concatenate([self.T_dist,
                                         self.dmp_dist]).tolist()
        rep.loc['std'] = np.concatenate([self.T_dist_std,
                                         self.dmp_dist_std]).tolist()

        self.soliddist = rep
        rep.to_excel(plib.Path(out_path, self.name + '_soliddist.xlsx'))
        self.soliddist_report_computed = True
        return self.soliddist

    # methods to plot results for a single sample
    def tg_plot(self, TG_lab='TG [wt%]', grid=False):
        if not self.proximate_computed:
            self.proximate_analysis()
        out_path = plib.Path(self.out_path, 'SingleSamplePlots')
        out_path.mkdir(parents=True, exist_ok=True)
        filename = self.name
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
        FigSave(filename + '_tg', out_path, fig, ax, axt, fig_par,
                xLab='time [min]',
                yLab=['T ['+self.T_symbol+']',
                      TG_lab+'(stb)', TG_lab+'(db)'], grid=grid)

    def dtg_plot(self, TG_lab='TG [wt%]', DTG_lab=None, grid=False):
        if not self.proximate_computed:
            self.proximate_analysis()
        out_path = plib.Path(self.out_path, 'SingleSamplePlots')
        out_path.mkdir(parents=True, exist_ok=True)

        if DTG_lab is None:
            if self.dtg_basis == 'temperature':
                DTG_lab = 'DTG [wt%/' + self.T_symbol + ']'
            elif self.dtg_basis == 'time':
                DTG_lab='DTG [wt%/min]'
        filename = self.name

        fig, ax, axt, fig_par = FigCreate(rows=3, cols=1, plot_type=0,
                                          paper_col=1, font=self.plot_font)
        for f in range(self.n_repl):
            ax[0].plot(self.time_dtg, self.T_dtg, color=clrs[f],
                       linestyle=lnstls[f], label=self.filenames[f])
            ax[1].plot(self.time_dtg, self.mp_db_dtg_stk[:, f], color=clrs[f],
                       linestyle=lnstls[f])
            ax[2].plot(self.time_dtg, self.dtg_db_stk[:, f], color=clrs[f],
                       linestyle=lnstls[f])
            if self.oxidation_computed:
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
        FigSave(filename + '_dtg', out_path, fig, ax, axt, fig_par,
                xLab='time [min]',
                yLab=['T ['+self.T_symbol+']', TG_lab + '(db)',
                      DTG_lab + '(db)'], grid=grid)

    def soliddist_plot(self, paper_col=1, hgt_mltp=1.25, TG_lab='TG [wt%]',
                       grid=False):
        # slightly different plotting behaviour (uses averages)
        if not self.soliddist_computed:
            self.soliddist_analysis()
        out_path = plib.Path(self.out_path, 'SingleSamplePlots')
        out_path.mkdir(parents=True, exist_ok=True)
        filename = self.name
        fig, ax, axt, fig_par = FigCreate(rows=2, cols=1, plot_type=0,
                                          paper_col=paper_col,
                                          hgt_mltp=hgt_mltp)

        ax[0].plot(self.time, self.T)
        ax[0].fill_between(self.time, self.T - self.T_std, self.T + self.T_std,
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
        FigSave(filename + '_soliddist', out_path, fig, ax, axt, fig_par,
                xLab='time [min]',
                yLab=['T ['+self.T_symbol+']', TG_lab+'(db)'],
                grid=grid)

    def deconv_plot(self, filename='Deconv',
                    xLim=None, yLim=None, grid=False, DTG_lab=None,
                    pdf=False, svg=False, legend='best'):
        if not self.deconv_computed:
            self.deconv_analysis()
        out_path_dcv = plib.Path(self.out_path, 'SingleSampleDeconvs')
        out_path_dcv.mkdir(parents=True, exist_ok=True)
        if DTG_lab is None:
            if self.dtg_basis == 'temperature':
                DTG_lab = 'DTG [wt%/' + self.T_symbol + ']'
            elif self.dtg_basis == 'time':
                DTG_lab='DTG [wt%/min]'
        filename = self.name
        fig, ax, axt, fig_par = FigCreate(rows=1, cols=1, plot_type=0,
                                          paper_col=0.78, hgt_mltp=1.25,
                                          )
        # Plot DTG data
        ax[0].plot(self.T_dtg, self.dtg_db, color='black', label='DTG')
        ax[0].fill_between(self.T_dtg, self.dtg_db - self.dtg_db_std,
                           self.dtg_db + self.dtg_db_std, color='black',
                           alpha=0.3)

        # Plot best fit and individual peaks
        ax[0].plot(self.T_dtg, self.dcv_best_fit, label='best fit', color='red',
                   linestyle='--')
        ax[0].fill_between(self.T_dtg,
                           self.dcv_best_fit - self.dcv_best_fit_std,
                           self.dcv_best_fit + self.dcv_best_fit_std,
                           color='red', alpha=0.3)
        clrs_p = clrsp = clrs[:3] + clrs[5:]  # avoid using red
        p = 0
        for peak, peak_std in zip(self.dcv_peaks.T, self.dcv_peaks_std.T):

            ax[0].plot(self.T_dtg, peak, label='peak ' + str(int(p+1)),
                       color=clrs_p[p], linestyle=lnstls[p])
            ax[0].fill_between(self.T_dtg,
                               peak - peak_std,
                               peak + peak_std,
                               color=clrs_p[p], alpha=0.3)
            p += 1
        ax[0].annotate(f"r$^2$={self.dcv_r2:.2f}", xycoords='axes fraction',
                       xy=(0.85, 0.96), size='x-small')

        # Save figure using FigSave
        FigSave(filename, out_path_dcv, fig, ax, axt, fig_par,
                xLab='T ['+self.T_symbol+']', yLab=DTG_lab,
                xLim=xLim, yLim=yLim, legend=legend,
                pdf=pdf, svg=svg)  # Set additional parameters as needed


# =============================================================================
# # functions to print reports with ave and std of multiple samples
# =============================================================================
def proximate_multi_report(exps, filename='Rep'):
    for exp in exps:
        if not exp.proximate_report_computed:
            exp.proximate_report()
    out_path = plib.Path(exps[0].out_path, 'MultiSampleReports')
    out_path.mkdir(parents=True, exist_ok=True)

    rep = pd.DataFrame(columns=list(exps[0].proximate))
    for exp in exps:
        rep.loc[exp.label + '_ave'] = exp.proximate.loc['ave', :]
    for exp in exps:
        rep.loc[exp.label + '_std'] = exp.proximate.loc['std', :]
    rep.to_excel(plib.Path(out_path, filename + '_prox.xlsx'))
    return rep


def oxidation_multi_report(exps, filename='Rep'):
    for exp in exps:
        if not exp.oxidation_report_computed:
            exp.oxidation_report()
    out_path = plib.Path(exps[0].out_path, 'MultiSampleReports')
    out_path.mkdir(parents=True, exist_ok=True)

    rep = pd.DataFrame(columns=list(exps[0].oxidation))
    for exp in exps:
        rep.loc[exp.label + '_ave'] = exp.oxidation.loc['ave', :]
    for exp in exps:
        rep.loc[exp.label + '_std'] = exp.oxidation.loc['std', :]
    rep.to_excel(plib.Path(out_path, filename + '_oxid.xlsx'))
    return rep


def soliddist_multi_report(exps, filename='Rep'):
    for exp in exps:
        if not exp.soliddist_report_computed:
            exp.soliddist_report()
    out_path = plib.Path(exps[0].out_path, 'MultiSampleReports')
    out_path.mkdir(parents=True, exist_ok=True)

    rep = pd.DataFrame(columns=list(exps[0].soliddist))
    for exp in exps:
        rep.loc[exp.label + '_ave'] = exp.soliddist.loc['ave', :]
    for exp in exps:
        rep.loc[exp.label + '_std'] = exp.soliddist.loc['std', :]
    rep.to_excel(plib.Path(out_path, filename + '_soliddist.xlsx'))
    return rep


# =============================================================================
# # functions for plotting ave and std of multiple samples
# =============================================================================
def tg_multi_plot(exps, filename='Fig', paper_col=.78, hgt_mltp=1.25,
                  xLim=None, yLim=[0, 100], yTicks=None, grid=False,
                  TG_lab='TG [wt%]', lttrs=False, pdf=False, svg=False):
    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    out_path_TGs = plib.Path(exps[0].out_path, 'MultiSamplePlots')
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
    FigSave(filename + '_tg', out_path_TGs, fig, ax, axt, fig_par,
            xLim=xLim, yLim=yLim,
            yTicks=yTicks,
            xLab='T [' + exps[0].T_symbol + ']', legend='upper right',
            yLab=TG_lab, annotate_lttrs=lttrs, grid=grid, pdf=pdf, svg=svg)


def dtg_multi_plot(exps, filename='Fig', paper_col=.78, hgt_mltp=1.25,
                   xLim=None, yLim=None, yTicks=None, grid=False,
                   DTG_lab=None, lttrs=False, plt_gc=False, gc_Tlim=300,
                   pdf=False, svg=False):
    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    out_path = plib.Path(exps[0].out_path, 'MultiSamplePlots')
    out_path.mkdir(parents=True, exist_ok=True)

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
        ax[0].vlines(gc_Tlim, ymin=yLim[0], ymax=yLim[1],
                     linestyle=lnstls[1], color=clrs[7],
                     label='T$_{max GC-MS}$')
    ax[0].legend(loc='lower right')
    FigSave(filename + '_dtg', out_path, fig, ax, axt, fig_par,
            xLim=xLim, yLim=yLim,
            yTicks=yTicks,
            yLab=DTG_lab, xLab='T [' + exps[0].T_symbol + ']',
            pdf=pdf, svg=svg, annotate_lttrs=lttrs, grid=grid)


def proximate_multi_plot(exps, filename="Prox",
                         smpl_labs=None, xlab_rot=0,
                         paper_col=.8, hgt_mltp=1.5, grid=False,
                         bboxtoanchor=True, x_anchor=1.13, y_anchor=1.02,
                         legend_loc='best', yLim=[0, 100], ytLim=[0, 1],
                         yTicks=None, ytTicks=None):
    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    out_path = plib.Path(exps[0].out_path, 'MultiSamplePlots')
    out_path.mkdir(parents=True, exist_ok=True)
    vars_bar = ['Moisture (stb)', 'VM (db)', 'FC (db)', 'Ash (db)']
    vars_scat = ['Mean TG dev.']
    if smpl_labs:
        labels = smpl_labs
    else:
        labels = [exp.label for exp in exps]
    df_ave = pd.DataFrame(columns=vars_bar, index=labels)
    df_std = pd.DataFrame(columns=vars_bar, index=labels)
    df_ave['Moisture (stb)'] = [exp.moist_ar for exp in exps]
    df_ave['VM (db)'] = [exp.vm_db for exp in exps]
    df_ave['FC (db)'] = [exp.fc_db for exp in exps]
    df_ave['Ash (db)'] = [exp.ash_db for exp in exps]
    df_std['Moisture (stb)'] = [exp.moist_ar_std for exp in exps]
    df_std['VM (db)'] = [exp.vm_db_std for exp in exps]
    df_std['FC (db)'] = [exp.fc_db_std for exp in exps]
    df_std['Ash (db)'] = [exp.ash_db_std for exp in exps]

    S_combs = [exp.AveTGstd_p for exp in exps]
    fig, ax, axt, fig_par = FigCreate(1, 1, 1, paper_col=paper_col,
                                      hgt_mltp=hgt_mltp)
    df_ave.plot(kind='bar', ax=ax[0], yerr=df_std, capsize=2, width=.85,
                ecolor='k', edgecolor='black', rot=xlab_rot)
    bars = ax[0].patches
    patterns = [None, '//', '...', '--']  # set hatch patterns in correct order
    hatches = []  # list for hatches in the order of the bars
    for h in patterns:  # loop over patterns to create bar-ordered hatches
        for i in range(int(len(bars) / len(patterns))):
            hatches.append(h)
    # loop over bars and hatches to set hatches in correct order
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axt[0].errorbar(x=df_ave.index, y=S_combs, linestyle='None',
                    marker=mrkrs[0], color=clrs[4], markersize=10,
                    markeredgecolor='k', label='Mean TG dev.')
    hnd_ax, lab_ax = ax[0].get_legend_handles_labels()
    hnd_axt, lab_axt = axt[0].get_legend_handles_labels()
    if bboxtoanchor:  # legend goes outside of plot area
        ax[0].legend(hnd_ax + hnd_axt, lab_ax + lab_axt,
                     loc='upper left', bbox_to_anchor=(x_anchor, y_anchor))
    else:  # legend is inside of plot area
        ax[0].legend(hnd_ax + hnd_axt, lab_ax + lab_axt,
                     loc=legend_loc)
    if xlab_rot != 0:
        ax[0].set_xticklabels(df_ave.index, rotation=xlab_rot, ha='right',
                              rotation_mode='anchor')
    FigSave(filename + '_prox', out_path, fig, ax, axt, fig_par, tight_layout=True,
            legend=None,
            yLab='mass fraction [wt%]', ytLab='Mean TG deviation [%]',
            yLim=yLim, ytLim=ytLim, yTicks=yTicks, ytTicks=ytTicks, grid=grid)


def oxidation_multi_plot(exps, filename="Oxidations",
                         smpl_labs=None, xlab_rot=0,
                         paper_col=.8, hgt_mltp=1.5, grid=False,
                         bboxtoanchor=True, x_anchor=1.13, y_anchor=1.02,
                         legend_loc='best',
                         yLim=None, ytLim=None, yTicks=None, ytTicks=None):
    for exp in exps:
        if not exp.oxidation_computed:
            exp.oxidation_analysis()
    out_path = plib.Path(exps[0].out_path, 'MultiSamplePlots')
    out_path.mkdir(parents=True, exist_ok=True)
    vars_bar = ['T$_i$', 'T$_p$', 'T$_b$']
    vars_scat = ['S (combustibility index)']
    if smpl_labs:
        labels = smpl_labs
    else:
        labels = [exp.label for exp in exps]
    df_ave = pd.DataFrame(columns=vars_bar, index=labels)
    df_std = pd.DataFrame(columns=vars_bar, index=labels)
    df_ave['T$_i$'] = [exp.Ti for exp in exps]
    df_ave['T$_p$'] = [exp.Tp for exp in exps]
    df_ave['T$_b$'] = [exp.Tb for exp in exps]
    df_std['T$_i$'] = [exp.Ti_std for exp in exps]
    df_std['T$_p$'] = [exp.Tp_std for exp in exps]
    df_std['T$_b$'] = [exp.Tb_std for exp in exps]

    S_combs = [exp.S for exp in exps]
    S_combs_std = [exp.S_std for exp in exps]
    fig, ax, axt, fig_par = FigCreate(1, 1, 1, paper_col=paper_col,
                                      hgt_mltp=hgt_mltp)
    df_ave.plot(kind='bar', ax=ax[0], yerr=df_std, capsize=2, width=.85,
                ecolor='k', edgecolor='black', rot=xlab_rot)
    bars = ax[0].patches
    patterns = [None, '//', '...']  # set hatch patterns in the correct order
    hatches = []  # list for hatches in the order of the bars
    for h in patterns:  # loop over patterns to create bar-ordered hatches
        for i in range(int(len(bars) / len(patterns))):
            hatches.append(h)
    # loop over bars and hatches to set hatches in correct order
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axt[0].errorbar(x=df_ave.index, y=S_combs, yerr=S_combs_std,
                    linestyle='None', marker=mrkrs[0], ecolor='k', capsize=2,
                    markeredgecolor='k', color=clrs[3], markersize=10,
                    label='S')
    hnd_ax, lab_ax = ax[0].get_legend_handles_labels()
    hnd_axt, lab_axt = axt[0].get_legend_handles_labels()
    if bboxtoanchor:  # legend goes outside of plot area
        ax[0].legend(hnd_ax + hnd_axt, lab_ax + lab_axt,
                     loc='upper left', bbox_to_anchor=(x_anchor, y_anchor))
    else:  # legend is inside of plot area
        ax[0].legend(hnd_ax + hnd_axt, lab_ax + lab_axt,
                     loc=legend_loc)
    if xlab_rot != 0:
        ax[0].set_xticklabels(df_ave.index, rotation=xlab_rot, ha='right',
                              rotation_mode='anchor')
    FigSave(filename + '_oxidation', out_path, fig, ax, axt, fig_par,
            tight_layout=True,
            legend=None, ytLab='S (combustion index) [-]',
            yLab='T [' + exps[0].T_symbol + ']',
            yLim=yLim, ytLim=ytLim, yTicks=yTicks, ytTicks=ytTicks, grid=grid)


def soliddist_multi_plot(exps, filename="Dist",
                         TG_lab='TG [wt%]', DTG_lab='DTG [wt%/min]',
                         hgt_mltp=1.25, paper_col=.78, labels=None, lttrs=False,
                         xLim=None, yLim=[[0, 1000], [0, 100]], yTicks=None,
                         print_dfs=True, grid=False):
    for exp in exps:
        if not exp.soliddist_computed:
            exp.soliddist_analysis()
    out_path = plib.Path(exps[0].out_path, 'MultiSamplePlots')
    out_path.mkdir(parents=True, exist_ok=True)
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
        # ax[1].legend(loc='center left')
    FigSave(filename + '_soliddist', out_path, fig, ax, axt, fig_par,
            xLim=xLim, yLim=yLim, yTicks=yTicks, xLab='time [min]',
            yLab=['T [' + exps[0].T_symbol + ']', TG_lab+'(db)'],
            annotate_lttrs=lttrs, grid=grid)


def cscd_multi_plot(exps, filename='Fig', paper_col=.78, hgt_mltp=1.25,
               xLim=None,clrs_cscd=False,  yLim0_cscd=-8,
               shifts_cscd=np.asarray([0, 11, 5, 10, 10, 10, 11]),
               peaks_cscd=None, peak_names=None, dh_names_cscd=0.1,
               loc_names_cscd=130,
               hgt_mltp_cscd=1.5, legend_cscd='lower right',
               y_values_cscd=[-10, 0],
               lttrs=False, DTG_lab=None, pdf=False, svg=False):
    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    out_path = plib.Path(exps[0].out_path, 'MultiSamplePlots')
    out_path.mkdir(parents=True, exist_ok=True)
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
                           exp.dtg_db[np.argmax(exp.T_dtg > loc_names_cscd)] +
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
    FigSave(filename + '_cscd', out_path, fig, ax, axt, fig_par,
            legend=legend_cscd, annotate_lttrs=lttrs,
            xLab='T [' + exps[0].T_symbol + ']', yLab=DTG_lab,
            xLim=xLim, yLim=yLim_cscd, svg=svg, pdf=pdf)


def KAS_analysis(exps, ramps, alpha=np.arange(0.05, .9, 0.05)):
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
        if exp.T_unit == 'Kelvin':
            TK = exp.T_dtg
        elif exp.T_unit == 'Celsius':
            TK = exp.T_dtg + 273.15
        mp_db_dtg = exp.mp_db_dtg
        mdaf = ((mp_db_dtg - np.min(mp_db_dtg))
                / (np.max(mp_db_dtg) - np.min(mp_db_dtg)))
        a = 1 - mdaf
        for i in range(n_alpha):
            v_a[i] = np.argmax(a > alpha[i])
        xmatr[:, e] = 1/TK[v_a]*1000
        for i in range(n_alpha):
            # BETA IS IN MINUTE HERE
            ymatr[i, e] = np.log(ramps[e]/TK[v_a[i]]**2)

    for i in range(n_alpha):
        p, cov = np.polyfit(xmatr[i, :], ymatr[i, :], 1, cov=True)
        v_fit.append(np.poly1d(p))
        v_res_fit = ymatr[i, :] - v_fit[i](xmatr[i, :])
        r_sqrd_fit[i] = (1 - (np.sum(v_res_fit**2)
                              / np.sum((ymatr[i, :]
                                        - np.mean(ymatr[i, :]))**2)))
        Ea[i] = -v_fit[i][1] * R_gas
        Ea_std[i] = np.sqrt(cov[0][0]) * R_gas
    # the name is obtained supposing its Sample-Ox# with # = rate
    if '-Ox' in exps[0].name:
        name = exps[0].name.split('-Ox')[0]
    else:
        name = exps[0].name.split('Ox')[0]
    kas = {'Ea': Ea, 'Ea_std': Ea_std, 'alpha': alpha, 'ramps': ramps,
           'xmatr': xmatr, 'ymatr': ymatr, 'v_fit': v_fit, 'name': name}
    for e, exp in enumerate(exps):
        exp.kas = kas
        exp.KAS_computed = True
    return kas


def KAS_plot_isolines(exps, kas_names=None, filename='KAsIso',
                      paper_col=.78, hgt_mltp=1.25, xLim=None, yLim=None,
                      annt_names=True, annotate_lttrs=False, leg_cols=1,
                      bboxtoanchor=True, x_anchor=1.13, y_anchor=1.02,
                      legend_loc='best'):
    for exp in exps:
        if not exp.KAS_computed:
            print('need KAS analysis')
    out_path = plib.Path(exps[0].out_path, 'KAS')
    out_path.mkdir(parents=True, exist_ok=True)
    kass = [exp.kas for exp in exps]
    xmatrs = [kas['xmatr'] for kas in kass]
    # for plots
    if kas_names is None:
        kas_names = [kas['name'] for kas in kass]

    n_exp = len(kass)  # number of feedstocks
    alphas = [kas['alpha'] for kas in kass]
    if not all(np.array_equal(alphas[0], a) for a in alphas):
        print('samples have been analyzed at different alphas')

    else:
        alpha = alphas[0]
    n_alpha = len(alpha)

    x = np.linspace(np.min([np.min(xmatr) for xmatr in xmatrs]),
                    np.max([np.max(xmatr) for xmatr in xmatrs]), 100)
    rows = n_exp
    cols = 1
    ax_for_legend = 0
    if n_exp > 3:
        rows = round(rows/2 + 0.01)
        cols += 1
        paper_col *= 1.5
        ax_for_legend += 1
    fig, ax, axt, fig_par = FigCreate(rows=rows, cols=cols, plot_type=0,
                                      paper_col=paper_col, hgt_mltp=hgt_mltp)
    for k, kas in enumerate(kass):
        ymaxiso = np.max(kas['ymatr'])
        yminiso = np.min(kas['ymatr'])
        for i in range(n_alpha):
            lab = r'$\alpha$=' + str(np.round(alpha[i], 2))
            xmin = np.argwhere(kas['v_fit'][i](x) < ymaxiso)[0][0]
            try:
                xmax = np.argwhere(kas['v_fit'][i](x) < yminiso)[0][0]
            except IndexError:
                xmax = 0
            newx = x[xmin: xmax]
            ax[k].plot(newx, kas['v_fit'][i](newx),
                       color=clrs[i], linestyle=lnstls[i])
            ax[k].plot(kas['xmatr'][i, :], kas['ymatr'][i, :], color=clrs[i],
                       linestyle='None', marker=mrkrs[i])
            ax[k].plot([], [], color=clrs[i], linestyle=lnstls[i],
                       marker=mrkrs[i], label=lab)
            hnd_ax, lab_ax = ax[k].get_legend_handles_labels()
        if annt_names:
            ax[k].annotate(kas_names[k], xycoords="axes fraction",
                           xy=(0, 0), rotation=0, size="small",
                           xytext=(0.05, 0.93))
    if bboxtoanchor:  # legend goes outside of plot area

        ax[ax_for_legend].legend(ncol=leg_cols, loc='upper left',
                                 bbox_to_anchor=(x_anchor, y_anchor))
    else:  # legend is inside of plot area
        ax[0].legend(ncol=leg_cols, loc=legend_loc)
    FigSave(filename + '_iso', out_path, fig, ax, axt, fig_par,
            xLim=xLim, yLim=yLim, xLab='1000/T [1/K]', legend=None,
            annotate_lttrs=annotate_lttrs, yLab=r'ln($\beta$/T$^{2}$)',
            tight_layout=False)


def KAS_plot_Ea(exps, kas_names=None, filename='KASEa',
                paper_col=.78, hgt_mltp=1.25, xLim=[.1, .8], yLim=[0, 300],
                yTicks=None, annt_names=True, annotate_lttrs=False, leg_cols=1,
                bboxtoanchor=True, x_anchor=1.13, y_anchor=2.02,
                grid=False, plot_type='scatter',
                legend_loc='best'):
    for exp in exps:
        if not exp.KAS_computed:
            exp.KAS_analysis()
    out_path = plib.Path(exps[0].out_path, 'KAS')
    out_path.mkdir(parents=True, exist_ok=True)
    kass = [exp.kas for exp in exps]
    if kas_names is None:
        kas_names = [kas['name'] for kas in kass]
    alphas = [kas['alpha'] for kas in kass]
    if not all(np.array_equal(alphas[0], a) for a in alphas):
        print('samples have been analyzed at different alphas')

    else:
        alpha = alphas[0]
    # plot activation energy
    fig, ax, axt, fig_par = FigCreate(rows=1, cols=1, plot_type=0,
                                      paper_col=paper_col, hgt_mltp=hgt_mltp)
    for k, kas in enumerate(kass):
        if plot_type == 'scatter':
            ax[0].errorbar(alpha, kas['Ea'], kas['Ea_std'], color='k',
                           linestyle='None', capsize=3, ecolor=clrs[k])
            ax[0].plot(alpha, kas['Ea'], color=clrs[k],
                       linestyle='None', marker=mrkrs[k],
                       label=kas_names[k])
        elif plot_type == 'line':
            ax[0].plot(alpha, kas['Ea'], color=clrs[k],
                       linestyle=lnstls[k], label=kas_names[k])
            ax[0].fill_between(alpha, kas['Ea'] - kas['Ea_std'],
                               kas['Ea'] + kas['Ea_std'], color=clrs[k],
                               alpha=.3)
    if len(exps) == 1:
        legend_loc = False
    else:
        if bboxtoanchor:  # legend goes outside of plot area
            ax[0].legend(ncol=leg_cols, loc='upper left',
                         bbox_to_anchor=(x_anchor, y_anchor))
        else:  # legend is inside of plot area
            ax[0].legend(ncol=leg_cols,
                         loc=legend_loc)
    FigSave(filename + '_Ea', out_path, fig, ax, axt, fig_par,
            xLim=xLim, yLim=yLim,
            legend=legend_loc, yTicks=yTicks, xLab=r'$\alpha$ [-]',
            yLab=r'$E_{a}$ [kJ/mol]', grid=grid)

# %%
if __name__ == "__main__":
    folder = '_test'
    P1 = tga_exp(folder=folder, name='P1',
                  filenames=['MIS_1', 'MIS_2', 'MIS_3'],
                  t_moist=38, t_VM=147, T_unit='Celsius')
    P2 = tga_exp(folder=folder, name='P2', load_skiprows=0,
                  filenames=['DIG10_1', 'DIG10_2', 'DIG10_3'],
                  t_moist=22, t_VM=98, T_unit='Celsius')
    Ox5 = tga_exp(folder=folder, name='Ox5',
                     filenames=['CLSOx5_1', 'CLSOx5_2', 'CLSOx5_3'],
                     t_moist=38, t_VM=None, T_unit='Celsius')
    Ox10 = tga_exp(folder=folder, name='Ox10', load_skiprows=8,
                      filenames=['CLSOx10_2', 'CLSOx10_3'],
                      t_moist=38, t_VM=None, T_unit='Celsius')
    Ox50 = tga_exp(folder=folder, name='Ox50', load_skiprows=8,
                      filenames=['CLSOx50_4', 'CLSOx50_5'],
                      t_moist=38, t_VM=None, T_unit='Celsius')
    SD1 = tga_exp(folder=folder, name='SDa',
                  filenames=['SDa_1', 'SDa_2', 'SDa_3'],
                  t_moist=38, t_VM=None, T_unit='Celsius')
    SD2 = tga_exp(folder=folder, name='SDb',
                  filenames=['SDb_1', 'SDb_2', 'SDb_3'],
                  t_moist=38, t_VM=None, T_unit='Celsius')
    #%% si
    a = P1.proximate_report()
    b = P2.proximate_report()
    c = Ox5.oxidation_report()
    d = Ox10.oxidation_report()
    e = Ox50.oxidation_report()
    f = SD1.soliddist_report()
    g = SD2.soliddist_report()
    # %%
    P1.deconv_analysis([280, 380])
    Ox5.deconv_analysis([310, 450, 500])
    # %%
    tg_multi_plot([P1, P2, Ox5, SD1], filename='P1P2Ox5SD1')
    dtg_multi_plot([P1, P2, Ox5, SD1], filename='P1P2Ox5SD1')
    h = proximate_multi_report([P1, P2, Ox5, SD1], filename='P1P2Ox5SD1')
    proximate_multi_plot([P1, P2, Ox5, SD1], filename='P1P2Ox5SD1',
                         bboxtoanchor=False)
    i = oxidation_multi_report([Ox5, Ox10, Ox50], filename='Ox5Ox10Ox50')
    oxidation_multi_plot([Ox5, Ox10, Ox50], yLim=[250, 400],
                         filename='Ox5Ox10Ox50')
    j = soliddist_multi_report([SD1, SD2], filename='SD1SD2')
    soliddist_multi_plot([SD1, SD2], filename='SD1SD2')

    #%%
    k = KAS_analysis([Ox5, Ox10, Ox50], [5, 10, 50])
    KAS_plot_isolines([Ox5], filename='Ox5Ox10Ox50')
    KAS_plot_Ea([Ox5], filename='Ox5Ox10Ox50')


