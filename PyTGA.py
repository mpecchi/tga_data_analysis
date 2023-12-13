# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:28:04 2022

@author: mp933
"""
import pathlib as plib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import savgol_filter as SavFil
from lmfit.models import GaussianModel, LinearModel

# list with colors
clrs = sns.color_palette('deep', 30)
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
         'o', 'p', 'q', 'r']
# list with markers for plotting
mrkrs = ["o", "v", "X", "s", "p", "^", "P", "<", ">", "*", "d", "1", "2", "3",
         "o", "v", "X", "s", "p", "^", "P", "<", ">", "*", "d", "1", "2", "3"]

def paths_create(subfolder=''):
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


def fig_create(rows=1, cols=1, plot_type=0, paper_col=1,
    hgt_mltp=1, font='Dejavu Sans',
    sns_style='ticks'):
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
    if font == 'Times':  # set Times New Roman as the plot font fot text
        # this may require the installation of the font package
        sns.set_style(sns_style, {'font.family': 'Times New Roman'})
    else:  # leave Dejavu Sans (default) as the plot font fot text
        sns.set_style(sns_style)
    # single or double column in paperthat the figure will occupy
    if cols > 2:  # numer of columns (thus of plots in the figure)
        raise ValueError('\n fig_create: cols>2 not supported')
    
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


def fig_save(filename, out_path, fig, lst_ax, lst_axt, fig_par,
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


class TGAExp:
    """
    Class representing a TGA Experiment.

    Attributes:
        folder (str): The folder name for the experiment.
        in_path (str): The input path for the experiment.
        out_path (str): The output path for the experiment.
        T_unit (str): The unit of temperature (Celsius or Kelvin).
        T_symbol (str): The symbol for temperature unit.
        plot_font (str): The font for plotting.
        plot_grid (bool): Flag indicating whether to plot grid or not.
        dtg_basis (str): The basis for computing DTG (temperature or time).
        resolution_T_dtg (int): The resolution of temperature for DTG computation.
        dtg_w_SavFil (int): The width parameter for Savitzky-Golay filter in DTG computation.

    Methods:
        __init__(self, name, filenames, load_skiprows=0, label=None, time_moist=38, time_vm=147, T_initial_C=40, Tlims_dtg_C=[120, 800], correct_ash_mg=None, correct_ash_fr=None, oxid_Tb_thresh=1):
            Initializes a TGAExp object with the given parameters.
        
        load_single_file(self, filename):
            Loads a single file for the experiment.
        
        load_files(self):
            Loads all the files for the experiment.
        
        proximate_analysis(self):
            Performs proximate analysis on the loaded files.
    """
    # Rest of the code...
import pandas as pd
import numpy as np
import pathlib as plib

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

    folder = '_test'
    in_path, out_path = paths_create(folder)
    T_unit='Celsius'
    if T_unit == 'Celsius':
        T_symbol = '°C'
    elif T_unit == 'Kelvin':
        T_symbol = 'K'
    plot_font='Dejavu Sans'
    plot_grid=False
    dtg_basis='temperature'
    resolution_T_dtg=5
    dtg_w_SavFil=101
    
    def __init__(self, name, filenames, load_skiprows=0,
                 label=None, time_moist=38, 
                 time_vm=147, T_initial_C=40, Tlims_dtg_C=[120, 800], 
                 correct_ash_mg=None, correct_ash_fr=None,
                 oxid_Tb_thresh=1):
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
        self.column_name_mapping = \
            {'Time': 't_min', 'Temperature': 'T_C', 'Weight': 'm_p',
             'Weight.1': 'm_mg', 'Heat Flow': 'heatflow_mW',
             '##Temp./>C': 'T_C', 'Time/min': 't_min', 'Mass/%': 'm_p',
             'Segment': 'segment'}
        self.time_moist = time_moist
        self.time_vm = time_vm
        self.correct_ash_mg = correct_ash_mg
        self.correct_ash_fr = correct_ash_fr
        self.T_initial_C = T_initial_C
        if TGAExp.T_unit == 'Celsius':
            self.Tlims_dtg = Tlims_dtg_C
        elif TGAExp.T_unit == 'Kelvin':
            self.Tlims_dtg = [T + 273.15 for T in Tlims_dtg_C]
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

    def load_single_file(self, filename):
        """
        Loads a single file for the experiment.

        Args:
            filename (str): The filename of the file to load.

        Returns:
            list: The list of loaded files.
        """
        path = plib.Path(TGAExp.in_path, filename + '.txt')
        if not path.is_file():
            path = plib.Path(TGAExp.in_path, filename + '.csv')
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
        """
        Loads all the files for the experiment.

        Returns:
            list: The list of loaded files.
        """
        print('\n' + self.name)
        # import files and makes sure that replicates have the same size
        files, len_files,  = [], []
        for filename in self.filenames:
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
            file = file[file['T_C'] >= self.T_initial_C].copy()
            file['T_K'] = file['T_C'] + 273.15
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
            if TGAExp.T_unit == 'Celsius':
                self.T_stk[:, f] = file['T_C']
            elif TGAExp.T_unit == 'Kelvin':
                self.T_stk[:, f] = file['T_K']
            self.time_stk[:, f] = file['t_min']

            self.m_ar_stk[:, f] = file['m_mg']
            self.mp_ar_stk[:, f] = file['m_p']

            self.idx_moist_stk[f] = np.argmax(self.time_stk[:, f]
                                              > self.time_moist+0.01)


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
            if self.time_vm is not None:
                self.idx_vm_stk[f] = np.argmax(self.time_stk[:, f] > self.time_vm)
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
                          )*TGAExp.resolution_T_dtg)
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
            if TGAExp.dtg_basis == 'temperature':
                dtg = np.gradient(self.mp_db_dtg_stk[:, f], self.T_dtg)
            if TGAExp.dtg_basis == 'time':
                dtg = np.gradient(self.mp_db_dtg_stk[:, f],
                                  self.time_dtg_stk[:, f])
            self.dtg_db_stk[:, f] = SavFil(dtg, TGAExp.dtg_w_SavFil, 1)
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
        self.dist_steps_min = steps_min + ['end']
        len_dist_step = len(self.dist_steps_min)

        self.T_dist_stk = np.ones((len_dist_step, self.n_repl))
        self.time_dist_stk = np.ones((len_dist_step, self.n_repl))
        self.dmp_dist_stk = np.ones((len_dist_step, self.n_repl))
        self.loc_dist_stk = np.ones((len_dist_step, self.n_repl))

        for f, file in enumerate(self.files):
            idxs = []
            for step in steps_min:
                idxs.append(np.argmax(self.time_stk[:, f] > step))
            idxs.append(len(self.time)-1)
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
        # Function implementation
        ...
    def deconv_analysis(self, centers, sigmas=None, amplitudes=None,
                        c_mins=None, c_maxs=None, s_mins=None, s_maxs=None,
                        a_mins=None, a_maxs=None, TLim=None):
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

            out_path = plib.Path(TGAExp.out_path, 'SingleSampleReports')
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
            out_path = plib.Path(TGAExp.out_path, 'SingleSampleReports')
            out_path.mkdir(parents=True, exist_ok=True)
            if TGAExp.T_unit == 'Celsius':
                TiTpTb = ['Ti_C', 'Tp_C', 'Tb_C']
            elif TGAExp.T_unit == 'Kelvin':
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
        out_path = plib.Path(TGAExp.out_path, 'SingleSampleReports')
        out_path.mkdir(parents=True, exist_ok=True)
        columns = ['T [' + TGAExp.T_symbol + '](' + str(s) + 'min)'
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
    def tg_plot(self, TG_lab='TG [wt%]'):
        """
        Plot the TGA data.

        Args:
            TG_lab (str, optional): Label for the TG axis. Defaults to 'TG [wt%]'.
        """
        if not self.proximate_computed:
            self.proximate_analysis()
        out_path = plib.Path(TGAExp.out_path, 'SingleSamplePlots')
        out_path.mkdir(parents=True, exist_ok=True)
        filename = self.name
        fig, ax, axt, fig_par = fig_create(rows=3, cols=1, plot_type=0,
                                          paper_col=1, font=TGAExp.plot_font)
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
        fig_save(filename + '_tg', out_path, fig, ax, axt, fig_par,
                xLab='time [min]',
                yLab=['T ['+TGAExp.T_symbol+']',
                      TG_lab+'(stb)', TG_lab+'(db)'], grid=TGAExp.plot_grid)

    def dtg_plot(self, TG_lab='TG [wt%]', DTG_lab=None):
            """
            Plot the DTG (Derivative Thermogravimetric) data.

            Args:
                TG_lab (str, optional): Label for TG data. Defaults to 'TG [wt%]'.
                DTG_lab (str, optional): Label for DTG data. Defaults to None.
            """
            
            if not self.proximate_computed:
                self.proximate_analysis()
            out_path = plib.Path(TGAExp.out_path, 'SingleSamplePlots')
            out_path.mkdir(parents=True, exist_ok=True)

            if DTG_lab is None:
                if TGAExp.dtg_basis == 'temperature':
                    DTG_lab = 'DTG [wt%/' + TGAExp.T_symbol + ']'
                elif TGAExp.dtg_basis == 'time':
                    DTG_lab='DTG [wt%/min]'
            filename = self.name

            fig, ax, axt, fig_par = fig_create(rows=3, cols=1, plot_type=0,
                                              paper_col=1, font=TGAExp.plot_font)
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
            fig_save(filename + '_dtg', out_path, fig, ax, axt, fig_par,
                    xLab='time [min]',
                    yLab=['T ['+TGAExp.T_symbol+']', TG_lab + '(db)',
                          DTG_lab + '(db)'], grid=TGAExp.plot_grid)

    def soliddist_plot(self, paper_col=1, hgt_mltp=1.25, TG_lab='TG [wt%]'):
            """
            Plot the solid distribution analysis.

            Args:
                paper_col (int): Number of columns in the plot for paper publication (default is 1).
                hgt_mltp (float): Height multiplier for the plot (default is 1.25).
                TG_lab (str): Label for the TG axis (default is 'TG [wt%]').

            Returns:
                None
            """
            
            # slightly different plotting behaviour (uses averages)
            if not self.soliddist_computed:
                self.soliddist_analysis()
            out_path = plib.Path(TGAExp.out_path, 'SingleSamplePlots')
            out_path.mkdir(parents=True, exist_ok=True)
            filename = self.name
            fig, ax, axt, fig_par = fig_create(rows=2, cols=1, plot_type=0,
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
            fig_save(filename + '_soliddist', out_path, fig, ax, axt, fig_par,
                    xLab='time [min]',
                    yLab=['T ['+TGAExp.T_symbol+']', TG_lab+'(db)'],
                    grid=TGAExp.plot_grid)

    def deconv_plot(self, filename='Deconv',
                    xLim=None, yLim=None, DTG_lab=None,
                    pdf=False, svg=False, legend='best'):
            """
            Plot the deconvolution results.

            Args:
                filename (str, optional): The filename to save the plot. Defaults to 'Deconv'.
                xLim (tuple, optional): The x-axis limits of the plot. Defaults to None.
                yLim (tuple, optional): The y-axis limits of the plot. Defaults to None.
                DTG_lab (str, optional): The label for the y-axis. Defaults to None.
                pdf (bool, optional): Whether to save the plot as a PDF file. Defaults to False.
                svg (bool, optional): Whether to save the plot as an SVG file. Defaults to False.
                legend (str, optional): The position of the legend in the plot. Defaults to 'best'.
            """
            
            if not self.deconv_computed:
                self.deconv_analysis()
            out_path_dcv = plib.Path(TGAExp.out_path, 'SingleSampleDeconvs')
            out_path_dcv.mkdir(parents=True, exist_ok=True)
            if DTG_lab is None:
                if TGAExp.dtg_basis == 'temperature':
                    DTG_lab = 'DTG [wt%/' + TGAExp.T_symbol + ']'
                elif TGAExp.dtg_basis == 'time':
                    DTG_lab='DTG [wt%/min]'
            filename = self.name
            fig, ax, axt, fig_par = fig_create(rows=1, cols=1, plot_type=0,
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

            # Save figure using fig_save
            fig_save(filename, out_path_dcv, fig, ax, axt, fig_par,
                    xLab='T ['+TGAExp.T_symbol+']', yLab=DTG_lab,
                    xLim=xLim, yLim=yLim, legend=legend, grid=TGAExp.plot_grid,
                    pdf=pdf, svg=svg)  # Set additional parameters as needed


# =============================================================================
# # functions to print reports with ave and std of multiple samples
# =============================================================================
def proximate_multi_report(exps, filename='Rep'):
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
                  xLim=None, yLim=[0, 100], yTicks=None, 
                  TG_lab='TG [wt%]', lttrs=False, pdf=False, svg=False):
    """
    Plot multiple thermogravimetric (TG) curves.

    Args:
        exps (list): List of experimental data objects.
        filename (str, optional): Name of the output file. Defaults to 'Fig'.
        paper_col (float, optional): Width of the figure in inches. Defaults to 0.78.
        hgt_mltp (float, optional): Height multiplier of the figure. Defaults to 1.25.
        xLim (tuple, optional): Limits of the x-axis. Defaults to None.
        yLim (list, optional): Limits of the y-axis. Defaults to [0, 100].
        yTicks (list, optional): Custom y-axis tick locations. Defaults to None.
        TG_lab (str, optional): Label for the y-axis. Defaults to 'TG [wt%]'.
        lttrs (bool, optional): Whether to annotate letters on the plot. Defaults to False.
        pdf (bool, optional): Whether to save the figure as a PDF file. Defaults to False.
        svg (bool, optional): Whether to save the figure as an SVG file. Defaults to False.

    Returns:
        None
    """
    
    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    out_path_TGs = plib.Path(exps[0].out_path, 'MultiSamplePlots')
    out_path_TGs.mkdir(parents=True, exist_ok=True)
    fig, ax, axt, fig_par = fig_create(rows=1, cols=1, plot_type=0,
                                      paper_col=paper_col,
                                      hgt_mltp=hgt_mltp)
    for i, exp in enumerate(exps):
        ax[0].plot(exp.T, exp.mp_db, color=clrs[i], linestyle=lnstls[i],
                   label=exp.label if exp.label else exp.name)
        ax[0].fill_between(exp.T, exp.mp_db - exp.mp_db_std,
                           exp.mp_db + exp.mp_db_std, color=clrs[i],
                           alpha=.3)
    fig_save(filename + '_tg', out_path_TGs, fig, ax, axt, fig_par,
            xLim=xLim, yLim=yLim,
            yTicks=yTicks,
            xLab='T [' + TGAExp.T_symbol + ']', legend='upper right',
            yLab=TG_lab, annotate_lttrs=lttrs, grid=TGAExp.plot_grid, pdf=pdf, svg=svg)


def dtg_multi_plot(exps, filename='Fig', paper_col=.78, hgt_mltp=1.25,
                   xLim=None, yLim=None, yTicks=None,
                   DTG_lab=None, lttrs=False, plt_gc=False, gc_Tlim=300,
                   pdf=False, svg=False):
    """
    Plot multiple DTG curves for a list of experiments.

    Args:
        exps (list): List of TGAExp objects representing the experiments.
        filename (str, optional): Name of the output file. Defaults to 'Fig'.
        paper_col (float, optional): Color of the plot background. Defaults to 0.78.
        hgt_mltp (float, optional): Height multiplier for the plot. Defaults to 1.25.
        xLim (tuple, optional): Limits of the x-axis. Defaults to None.
        yLim (tuple, optional): Limits of the y-axis. Defaults to None.
        yTicks (list, optional): Custom y-axis tick labels. Defaults to None.
        DTG_lab (str, optional): Label for the y-axis. Defaults to None.
        lttrs (bool, optional): Whether to annotate letters on the plot. Defaults to False.
        plt_gc (bool, optional): Whether to plot the GC-MS maximum temperature line. Defaults to False.
        gc_Tlim (int, optional): Maximum temperature for the GC-MS line. Defaults to 300.
        pdf (bool, optional): Whether to save the plot as a PDF file. Defaults to False.
        svg (bool, optional): Whether to save the plot as an SVG file. Defaults to False.

    Returns:
        None
    """
    
    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    out_path = plib.Path(exps[0].out_path, 'MultiSamplePlots')
    out_path.mkdir(parents=True, exist_ok=True)

    if not DTG_lab:
        DTG_lab = 'DTG [wt%/' + TGAExp.T_symbol + ']'

    fig, ax, axt, fig_par = fig_create(rows=1, cols=1, plot_type=0,
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
    fig_save(filename + '_dtg', out_path, fig, ax, axt, fig_par,
            xLim=xLim, yLim=yLim,
            yTicks=yTicks,
            yLab=DTG_lab, xLab='T [' + TGAExp.T_symbol + ']',
            pdf=pdf, svg=svg, annotate_lttrs=lttrs, grid=TGAExp.plot_grid)


def proximate_multi_plot(exps, filename="Prox",
                         smpl_labs=None, xlab_rot=0,
                         paper_col=.8, hgt_mltp=1.5,
                         bboxtoanchor=True, x_anchor=1.13, y_anchor=1.02,
                         legend_loc='best', yLim=[0, 100], ytLim=[0, 1],
                         yTicks=None, ytTicks=None):
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
    - yLim (list): Y-axis limits for the bar plot (default: [0, 100]).
    - ytLim (list): Y-axis limits for the scatter plot (default: [0, 1]).
    - yTicks (list): Y-axis tick positions for the bar plot (default: None).
    - ytTicks (list): Y-axis tick positions for the scatter plot (default: None).

    Returns:
    - None
    """
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
    fig, ax, axt, fig_par = fig_create(1, 1, 1, paper_col=paper_col,
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
    fig_save(filename + '_prox', out_path, fig, ax, axt, fig_par, tight_layout=True,
            legend=None,
            yLab='mass fraction [wt%]', ytLab='Mean TG deviation [%]',
            yLim=yLim, ytLim=ytLim, yTicks=yTicks, ytTicks=ytTicks, grid=TGAExp.plot_grid)


def oxidation_multi_plot(exps, filename="Oxidations",
                         smpl_labs=None, xlab_rot=0,
                         paper_col=.8, hgt_mltp=1.5,
                         bboxtoanchor=True, x_anchor=1.13, y_anchor=1.02,
                         legend_loc='best',
                         yLim=None, ytLim=None, yTicks=None, ytTicks=None):
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
    - yLim (tuple): Limits of the y-axis (default: None).
    - ytLim (tuple): Limits of the twin y-axis (default: None).
    - yTicks (list): Custom tick positions for the y-axis (default: None).
    - ytTicks (list): Custom tick positions for the twin y-axis (default: None).

    Returns:
    - None
    """

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
    fig, ax, axt, fig_par = fig_create(1, 1, 1, paper_col=paper_col,
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
    fig_save(filename + '_oxidation', out_path, fig, ax, axt, fig_par,
            tight_layout=True,
            legend=None, ytLab='S (combustion index) [-]',
            yLab='T [' + TGAExp.T_symbol + ']',
            yLim=yLim, ytLim=ytLim, yTicks=yTicks, ytTicks=ytTicks, grid=TGAExp.plot_grid)


def soliddist_multi_plot(exps, filename="Dist",
                         TG_lab='TG [wt%]', DTG_lab='DTG [wt%/min]',
                         hgt_mltp=1.25, paper_col=.78, labels=None, lttrs=False,
                         xLim=None, yLim=[[0, 1000], [0, 100]], yTicks=None,
                         print_dfs=True):
    """
    Plot the solid distribution for multiple experiments.

    Args:
        exps (list): List of Experiment objects.
        filename (str, optional): Name of the output file. Defaults to "Dist".
        TG_lab (str, optional): Label for the TG axis. Defaults to 'TG [wt%]'.
        DTG_lab (str, optional): Label for the DTG axis. Defaults to 'DTG [wt%/min]'.
        hgt_mltp (float, optional): Height multiplier for the plot. Defaults to 1.25.
        paper_col (float, optional): Color of the plot background. Defaults to .78.
        labels (list, optional): List of labels for the experiments. Defaults to None.
        lttrs (bool, optional): Whether to annotate letters on the plot. Defaults to False.
        xLim (list, optional): Limits for the x-axis. Defaults to None.
        yLim (list, optional): Limits for the y-axis. Defaults to [[0, 1000], [0, 100]].
        yTicks (list, optional): Custom tick locations for the y-axis. Defaults to None.
        print_dfs (bool, optional): Whether to print the dataframes. Defaults to True.
    """
    
    for exp in exps:
        if not exp.soliddist_computed:
            exp.soliddist_analysis()
    out_path = plib.Path(exps[0].out_path, 'MultiSamplePlots')
    out_path.mkdir(parents=True, exist_ok=True)
    if not labels:  # try with labels and use name if no label is given
        labels = [exp.label if exp.label else exp.name for exp in exps]
    fig, ax, axt, fig_par = fig_create(rows=2, cols=1, plot_type=0,
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
    fig_save(filename + '_soliddist', out_path, fig, ax, axt, fig_par,
            xLim=xLim, yLim=yLim, yTicks=yTicks, xLab='time [min]',
            yLab=['T [' + TGAExp.T_symbol + ']', TG_lab+'(db)'],
            annotate_lttrs=lttrs, grid=TGAExp.plot_grid)


def cscd_multi_plot(exps, filename='Fig', paper_col=.78, hgt_mltp=1.25,
               xLim=None,clrs_cscd=False,  yLim0_cscd=-8,
               shifts_cscd=np.asarray([0, 11, 5, 10, 10, 10, 11]),
               peaks_cscd=None, peak_names=None, dh_names_cscd=0.1,
               loc_names_cscd=130,
               hgt_mltp_cscd=1.5, legend_cscd='lower right',
               y_values_cscd=[-10, 0],
               lttrs=False, DTG_lab=None, pdf=False, svg=False):
    """
    Generate a cascaded multi-plot for a list of experiments.

    Args:
        exps (list): List of experiments.
        filename (str, optional): Filename for the saved plot. Defaults to 'Fig'.
        paper_col (float, optional): Width of the plot in inches. Defaults to 0.78.
        hgt_mltp (float, optional): Height multiplier for the plot. Defaults to 1.25.
        xLim (tuple, optional): X-axis limits for the plot. Defaults to None.
        clrs_cscd (bool, optional): Flag to use custom colors for each experiment. Defaults to False.
        yLim0_cscd (int, optional): Starting value for the y-axis limits. Defaults to -8.
        shifts_cscd (ndarray, optional): Array of shift values for each experiment. Defaults to np.asarray([0, 11, 5, 10, 10, 10, 11]).
        peaks_cscd (list, optional): List of peaks for each experiment. Defaults to None.
        peak_names (list, optional): List of names for each peak. Defaults to None.
        dh_names_cscd (float, optional): Shift value for peak names. Defaults to 0.1.
        loc_names_cscd (int, optional): Location for annotating experiment names. Defaults to 130.
        hgt_mltp_cscd (float, optional): Height multiplier for the cascaded plot. Defaults to 1.5.
        legend_cscd (str, optional): Location of the legend. Defaults to 'lower right'.
        y_values_cscd (list, optional): List of y-axis tick values. Defaults to [-10, 0].
        lttrs (bool, optional): Flag to annotate letters for each experiment. Defaults to False.
        DTG_lab (str, optional): Label for the y-axis. Defaults to None.
        pdf (bool, optional): Flag to save the plot as a PDF file. Defaults to False.
        svg (bool, optional): Flag to save the plot as an SVG file. Defaults to False.

    Returns:
        None
    """
    for exp in exps:
        if not exp.proximate_computed:
            exp.proximate_analysis()
    out_path = plib.Path(exps[0].out_path, 'MultiSamplePlots')
    out_path.mkdir(parents=True, exist_ok=True)
    labels = [exp.label if exp.label else exp.name for exp in exps]
    if not DTG_lab:
        DTG_lab = 'DTG [wt%/' + TGAExp.T_symbol + ']'
    yLim_cscd = [yLim0_cscd, np.sum(shifts_cscd)]
    dh = np.cumsum(shifts_cscd)
    fig, ax, axt, fig_par = fig_create(1, 1, paper_col=.78,
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
    fig_save(filename + '_cscd', out_path, fig, ax, axt, fig_par,
            legend=legend_cscd, annotate_lttrs=lttrs,
            xLab='T [' + TGAExp.T_symbol + ']', yLab=DTG_lab,
            xLim=xLim, yLim=yLim_cscd, svg=svg, pdf=pdf, grid=TGAExp.plot_grid)


def KAS_analysis(exps, ramps, alpha=np.arange(0.05, .9, 0.05)):
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
    """
    Plot isolines for KAS analysis.

    Parameters:
    - exps (list): List of experiments.
    - kas_names (list, optional): List of names for each KAS analysis. If not provided, the names will be extracted from the KAS analysis data.
    - filename (str, optional): Name of the output file. Default is 'KAsIso'.
    - paper_col (float, optional): Width of the plot in inches. Default is 0.78.
    - hgt_mltp (float, optional): Height multiplier for the plot. Default is 1.25.
    - xLim (tuple, optional): Limits for the x-axis. Default is None.
    - yLim (tuple, optional): Limits for the y-axis. Default is None.
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
    fig, ax, axt, fig_par = fig_create(rows=rows, cols=cols, plot_type=0,
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
    fig_save(filename + '_iso', out_path, fig, ax, axt, fig_par,
            xLim=xLim, yLim=yLim, xLab='1000/T [1/K]', legend=None,
            annotate_lttrs=annotate_lttrs, yLab=r'ln($\beta$/T$^{2}$)',
            tight_layout=False, grid=TGAExp.plot_grid)


def KAS_plot_Ea(exps, kas_names=None, filename='KASEa',
                paper_col=.78, hgt_mltp=1.25, xLim=[.1, .8], yLim=[0, 300],
                yTicks=None, annt_names=True, annotate_lttrs=False, leg_cols=1,
                bboxtoanchor=True, x_anchor=1.13, y_anchor=2.02,
                plot_type='scatter',
                legend_loc='best'):
    """
    Plot the activation energy (Ea) for multiple experiments.

    Parameters:
    - exps (list): List of experiments.
    - kas_names (list, optional): List of names for the KAS analysis. If not provided, the names will be extracted from the experiments.
    - filename (str, optional): Name of the output file. Default is 'KASEa'.
    - paper_col (float, optional): Color of the plot background. Default is 0.78.
    - hgt_mltp (float, optional): Height multiplier for the plot. Default is 1.25.
    - xLim (list, optional): Limits of the x-axis. Default is [0.1, 0.8].
    - yLim (list, optional): Limits of the y-axis. Default is [0, 300].
    - yTicks (list, optional): Custom y-axis tick locations. Default is None.
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
    fig, ax, axt, fig_par = fig_create(rows=1, cols=1, plot_type=0,
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
    fig_save(filename + '_Ea', out_path, fig, ax, axt, fig_par,
            xLim=xLim, yLim=yLim,
            legend=legend_loc, yTicks=yTicks, xLab=r'$\alpha$ [-]',
            yLab=r'$E_{a}$ [kJ/mol]', grid=TGAExp.plot_grid)

# %%
if __name__ == "__main__":
    TGAExp.folder = '_test'
    TGAExp.plot_grid = False
    P1 = TGAExp(name='P1', filenames=['MIS_1', 'MIS_2', 'MIS_3'],
                time_moist=38, time_vm=147)
    P2 = TGAExp(name='P2', load_skiprows=0,
                filenames=['DIG10_1', 'DIG10_2', 'DIG10_3'],
                  time_moist=22, time_vm=98)
    Ox5 = TGAExp(name='Ox5',
                 filenames=['CLSOx5_1', 'CLSOx5_2', 'CLSOx5_3'],
                     time_moist=38, time_vm=None)
    Ox10 = TGAExp(name='Ox10', load_skiprows=8,
                      filenames=['CLSOx10_2', 'CLSOx10_3'],
                      time_moist=38, time_vm=None)
    Ox50 = TGAExp(name='Ox50', load_skiprows=8,
                      filenames=['CLSOx50_4', 'CLSOx50_5'],
                      time_moist=38, time_vm=None)
    SD1 = TGAExp(name='SDa',
                  filenames=['SDa_1', 'SDa_2', 'SDa_3'],
                  time_moist=38, time_vm=None)
    SD2 = TGAExp(name='SDb',
                  filenames=['SDb_1', 'SDb_2', 'SDb_3'],
                  time_moist=38, time_vm=None)
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


