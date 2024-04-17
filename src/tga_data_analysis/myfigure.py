from __future__ import annotations
import string
import pathlib as plib
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd


letters: list[str] = list(string.ascii_lowercase)

# list with colors
colors: list[tuple] = sns.color_palette("deep", 30)

# list with linestyles for plotting
linestyles: list[tuple] = [
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
markers: list[str] = [
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

hatches: list[str] = [
    None,
    "//",
    "...",
    "--",
    "O",
    "\\\\",
    "oo",
    "\\\\\\",
    "/////",
    ".....",
    "//",
    "...",
    "--",
    "O",
    "\\\\",
    "oo",
    "\\\\\\",
    "/////",
    ".....",
    "//",
    "...",
    "--",
    "O",
    "\\\\",
    "oo",
    "\\\\\\",
    "/////",
    ".....",
    "//",
    "...",
    "--",
    "O",
    "\\\\",
    "oo",
    "\\\\\\",
    "/////",
    ".....",
]


class MyFigure:
    """
    A class for creating and customizing figures using matplotlib and seaborn.

    :ivar broad_props: Properties applied to all axes.
    :type broad_props: dict
    :ivar kwargs: Configuration keyword arguments.
    :type kwargs: dict
    :ivar fig: The main figure object from matplotlib.
    :type fig: matplotlib.figure.Figure
    :ivar axs: Axes objects for the subplots.
    :type axs: list[matplotlib.axes.Axes]
    :ivar axts: Twin axes objects, if 'twinx' is enabled.
    :type axts: list[matplotlib.axes.Axes] or None
    :ivar n_axs: Number of axes/subplots.
    :type n_axs: int
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize a MyFigure object with optional configuration.

        :param kwargs: Configuration options as keyword arguments.
        :type kwargs: Any
        """

        self.kwargs = self.default_kwargs()
        self.kwargs.update(kwargs)  # Override defaults with any kwargs provided
        self.process_kwargs()

        self.create_figure()

        self.broad_props = self.broadcast_all_kwargs()  # broadcasted properties for each axis

        sns.set_palette(self.kwargs["color_palette"], self.kwargs["color_palette_n_colors"])
        sns.set_style(self.kwargs["sns_style"], {"font.family": self.kwargs["text_font"]})
        plt.rcParams.update({"font.size": self.kwargs["text_font_size"]})

        self.update_axes_props_pre_data()

    def default_kwargs(self) -> Dict[str, Any]:
        """
        Define default keyword arguments for the figure.

        :return: Default configuration settings.
        :rtype: Dict[str, Any]
        """
        defaults = {
            "filename": None,
            "out_path": None,
            "rows": 1,
            "cols": 1,
            "width": 6.0,
            "height": 6.0,
            "x_lab": None,
            "y_lab": None,
            "x_lim": None,
            "y_lim": None,
            "x_ticks": None,
            "y_ticks": None,
            "x_ticklabels": None,
            "y_ticklabels": None,
            "x_ticklabels_rotation": 0,
            "twinx": None,
            "yt_lab": None,
            "yt_lim": None,
            "yt_ticks": None,
            "yt_ticklabels": None,
            "legend": True,
            "legend_loc": "best",
            "legend_ncols": 1,
            "legend_title": None,
            "legend_bbox_xy": None,
            "annotate_letters": False,
            "annotate_letters_xy": (-0.15, -0.15),
            "annotate_letters_font_size": 10,
            "grid": None,
            "color_palette": "deep",
            "color_palette_n_colors": None,
            "text_font": "Dejavu Sans",
            "sns_style": "ticks",
            "text_font_size": 10,
            "legend_font_size": 10,
            "x_labelpad": 1,
            "y_labelpad": 1,
            "legend_borderpad": 0.3,
            "legend_handlelength": 1.5,
            "auto_apply_hatches_to_bars": True,
            "annotate_outliers": False,
            "annotate_outliers_decimal_places": 2,
            "mask_insignificant_data": False,
            "mask_insignificant_data_alpha": 0.3,
        }
        return defaults

    def process_kwargs(self) -> None:
        """
        Process and validate keyword arguments.

        Raises a ValueError if an invalid keyword argument is provided.
        """
        valid_kwargs = set(self.default_kwargs().keys())

        # Check for any invalid keyword arguments
        for kwarg in self.kwargs:
            if kwarg not in valid_kwargs:
                raise ValueError(f"Invalid keyword argument: '{kwarg}' \n {valid_kwargs = }")

        if self.kwargs["out_path"] is not None:
            self.kwargs["out_path"] = plib.Path(self.kwargs["out_path"])

        if self.kwargs["filename"] is not None:
            if not isinstance(self.kwargs["filename"], str):
                raise ValueError("filename must be a str.")

        self.kwargs["rows"] = int(self.kwargs["rows"])
        self.kwargs["cols"] = int(self.kwargs["cols"])
        self.kwargs["width"] = float(self.kwargs["width"])
        self.kwargs["height"] = float(self.kwargs["height"])
        self.kwargs["legend_ncols"] = int(self.kwargs["legend_ncols"])

        if self.kwargs["rows"] <= 0:
            raise ValueError("Number of rows must be positive.")
        if self.kwargs["cols"] <= 0:
            raise ValueError("Number of cols must be positive.")
        if self.kwargs["width"] <= 0:
            raise ValueError("Width must be positive.")
        if self.kwargs["height"] <= 0:
            raise ValueError("Height must be positive.")
        if self.kwargs["legend_ncols"] <= 0:
            raise ValueError("Number of legend columns must be positive.")

    def broadcast_all_kwargs(self) -> None:
        """
        Broadcast each kwargs property to a list of the same len as the axs or axts.
        If one value is given, it is repeated for all axes, if multiple values are
        given as a list, the list is maintained.
        This is performed on single value properties (labels, booleans) and on
        properties that are given as a list (x_lim, y_lim etc)
        """
        broad_props: dict[str, list] = {}
        # single props (one value per axis)
        for sprop in [
            "x_lab",
            "y_lab",
            "yt_lab",
            "grid",
            "x_ticklabels_rotation",
            "legend",
            "legend_loc",
            "legend_ncols",
            "legend_title",
            "annotate_outliers",
            "annotate_outliers_decimal_places",
            "annotate_letters",
            "mask_insignificant_data",
        ]:
            broad_props[sprop] = _broadcast_value_prop(self.kwargs[sprop], sprop, self.n_axs)
        # list props (a list per axis)
        for lprop in [
            "x_lim",
            "y_lim",
            "yt_lim",
            "x_ticks",
            "y_ticks",
            "yt_ticks",
            "x_ticklabels",
            "y_ticklabels",
            "yt_ticklabels",
            "legend_bbox_xy",
            "annotate_letters_xy",
        ]:
            broad_props[lprop] = _broadcast_list_prop(self.kwargs[lprop], lprop, self.n_axs)
        return broad_props

    def update_axes_props_pre_data(self) -> None:
        """
        Update properties that are applied to each axis individually before the axes are
        populated with data, as the data does not influence the behavior.
        Examples are axis labels or limits.
        """

        # Update each axis with the respective properties
        for i, ax in enumerate(self.axs):
            if self.broad_props["x_lab"][i] is not None:
                ax.set_xlabel(self.broad_props["x_lab"][i], labelpad=self.kwargs["x_labelpad"])
            if self.broad_props["y_lab"][i] is not None:
                ax.set_ylabel(self.broad_props["y_lab"][i], labelpad=self.kwargs["y_labelpad"])
            if self.broad_props["grid"][i] is not None:
                ax.grid(self.broad_props["grid"][i])
            if self.broad_props["x_lim"][i] is not None:
                ax.set_xlim(_adjust_lims(self.broad_props["x_lim"][i]))
            if self.broad_props["y_lim"][i] is not None:
                ax.set_ylim(_adjust_lims(self.broad_props["y_lim"][i]))
            if self.broad_props["x_ticks"][i] is not None:
                ax.set_xticks(self.broad_props["x_ticks"][i])
            if self.broad_props["y_ticks"][i] is not None:
                ax.set_yticks(self.broad_props["y_ticks"][i])
            if self.broad_props["x_ticklabels"][i] is not None:
                ax.set_xticklabels(self.broad_props["x_ticklabels"][i])
            if self.broad_props["y_ticklabels"][i] is not None:
                ax.set_yticklabels(self.broad_props["y_ticklabels"][i])

        if self.kwargs["twinx"]:
            for i, axt in enumerate(self.axts):
                if self.broad_props["yt_lab"][i] is not None:
                    axt.set_ylabel(self.broad_props["yt_lab"][i])
                if self.broad_props["yt_lim"][i] is not None:
                    axt.set_ylim(_adjust_lims(self.broad_props["yt_lim"][i]))
                if self.broad_props["yt_ticks"][i] is not None:
                    axt.set_yticks(self.broad_props["yt_ticks"][i])
                if self.broad_props["yt_ticklabels"][i] is not None:
                    axt.set_yticklabels(self.broad_props["yt_ticklabels"][i])

    def create_figure(self) -> MyFigure:
        """
        Creates the figure and its axes.

        :return: MyFigure
        :rtype: MyFigure
        """
        self.fig: Figure
        self.axs: Axes
        self.axts: Axes | None = None
        self.fig, axes = plt.subplots(
            self.kwargs["rows"],
            self.kwargs["cols"],
            figsize=(self.kwargs["width"], self.kwargs["height"]),
            constrained_layout=True,
        )
        # Ensure ax is always an array, even if it's just one subplot
        self.axs: list[Axes] = np.atleast_1d(axes).flatten().tolist()
        if self.kwargs["twinx"]:
            self.axts: list[Axes] = [a.twinx() for a in self.axs]

        self.n_axs = len(self.axs)
        return self

    def update_axes_props_post_data(self) -> None:
        """
        Update properties that are applied to each axis individually AFTER the axes are
        populated with data, as the data does DO influence the behavior.
        Examples are legend or annotating outliers or masking data.
        """
        for i, ax in enumerate(self.axs):
            if self.kwargs["auto_apply_hatches_to_bars"]:
                _apply_hatch_patterns_to_ax(ax)
            if self.broad_props["annotate_outliers"][i]:
                _annotate_outliers_to_ax(
                    ax, self.broad_props["annotate_outliers_decimal_places"][i]
                )
            if self.broad_props["x_ticklabels_rotation"][i] is not None:
                _rotate_x_labels_ax(ax, self.broad_props["x_ticklabels_rotation"][i])
            if self.broad_props["annotate_letters"][i]:
                _annotate_letters_to_ax(
                    ax,
                    letter=self.broad_props["annotate_letters"][i],
                    xy=self.broad_props["annotate_letters_xy"][i],
                    font_size=self.kwargs["annotate_letters_font_size"],
                )
            if self.broad_props["mask_insignificant_data"][i]:
                _mask_insignificant_data_in_ax(
                    ax, alpha=self.kwargs["mask_insignificant_data_alpha"]
                )

        if self.kwargs["twinx"]:
            for i, axt in enumerate(self.axts):
                if self.kwargs["auto_apply_hatches_to_bars"]:
                    _apply_hatch_patterns_to_ax(axt)
                if self.broad_props["annotate_outliers"][i]:
                    _annotate_outliers_to_ax(
                        axt, self.broad_props["annotate_outliers_decimal_places"][i]
                    )
                if self.broad_props["mask_insignificant_data"][i]:
                    _mask_insignificant_data_in_ax(
                        axt, alpha=self.kwargs["mask_insignificant_data_alpha"]
                    )

        for i, ax in enumerate(self.axs):
            if self.kwargs["twinx"]:
                axt = self.axts[i]
            else:
                axt = None
            if self.broad_props["legend"][i]:
                _add_legend_to_ax(
                    ax,
                    axt,
                    loc=self.broad_props["legend_loc"][i],
                    ncol=self.broad_props["legend_ncols"][i],
                    title=self.broad_props["legend_title"][i],
                    bbox_xy=self.broad_props["legend_bbox_xy"][i],
                    font_size=self.kwargs["legend_font_size"],
                    borderpad=self.kwargs["legend_borderpad"],
                    handlelength=self.kwargs["legend_handlelength"],
                    masked_values=self.broad_props["mask_insignificant_data"][i],
                )

    def save_figure(
        self,
        filename: str | None = None,
        out_path: plib.Path | None = None,
        tight_layout: bool = True,
        save_as_png: bool = True,
        save_as_pdf: bool = False,
        save_as_svg: bool = False,
        save_as_eps: bool = False,
        save_as_tif: bool = False,
        png_transparency: bool = False,
        dpi: int = 300,
        update_all_axis_props: bool = True,
    ) -> None:
        """
        Save the figure to a file.

        :param filename: The name of the file.
        :type filename: str | None
        :param out_path: The path to save the file.
        :type out_path: pathlib.Path | None
        :param tight_layout: Whether to use a tight layout.
        :type tight_layout: bool
        :param save_as_png: Save as PNG.
        :type save_as_png: bool
        :param save_as_pdf: Save as PDF.
        :type save_as_pdf: bool
        :param save_as_svg: Save as SVG.
        :type save_as_svg: bool
        :param save_as_eps: Save as EPS.
        :type save_as_eps: bool
        :param png_transparency: PNG transparency.
        :type png_transparency: bool
        """
        if update_all_axis_props:
            self.update_axes_props_post_data()
            self.fig.align_labels()  # align labels of subplots, needed only for multi plot
        # Saving the figure
        formats = {
            "png": save_as_png,
            "pdf": save_as_pdf,
            "svg": save_as_svg,
            "eps": save_as_eps,
            "tif": save_as_tif,
        }
        if filename is None:
            filename = self.kwargs["filename"]
        if out_path is None:
            out_path = self.kwargs["out_path"]

        for fmt, should_save in formats.items():
            if should_save:
                full_path = plib.Path(out_path, f"{filename}.{fmt}")
                self.fig.savefig(
                    full_path,
                    dpi=dpi,
                    transparent=png_transparency,
                    bbox_inches="tight" if tight_layout else None,
                )


def create_inset(
    ax: Axes,
    x_loc: tuple[float],
    y_loc: tuple[float],
    x_lim: tuple[float] | None = None,
    y_lim: tuple[float] | None = None,
) -> Axes:
    """
    Create an inset plot within an existing axis.

    :param ax: The parent axis.
    :type ax: Axes
    :param x_loc: X location for the inset.
    :type x_loc: tuple[float, float]
    :param y_loc: Y location for the inset.
    :type y_loc: tuple[float, float]
    :param x_lim: X limits for the inset.
    :type x_lim: tuple[float, float] | None
    :param y_lim: Y limits for the inset.
    :type y_lim: tuple[float, float] | None
    :return: The inset axes.
    :rtype: Axes
    """
    wdt = x_loc[1] - x_loc[0]
    hgt = y_loc[1] - y_loc[0]
    inset = ax.inset_axes([x_loc[0], y_loc[0], wdt, hgt])
    if x_lim is not None:
        inset.set_xlim(_adjust_lims(x_lim))
    if y_lim is not None:
        inset.set_ylim(_adjust_lims(y_lim))
    return inset


def _adjust_lims(lims: tuple[float] | None, gap=0.05) -> tuple[float] | None:
    """
    Adjust axis limits with a specified gap.

    :param lims: Axis limits to adjust.
    :type lims: tuple[float, float] | None
    :param gap: Percentage gap to add to the limits.
    :type gap: float, optional
    :return: Adjusted axis limits.
    :rtype: tuple[float, float] | None
    """
    if lims is None:
        return None
    else:
        new_lims = (
            lims[0] * (1 + gap) - gap * lims[1],
            lims[1] * (1 + gap) - gap * lims[0],
        )
        return new_lims


def _add_legend_to_ax(
    ax: Axes,
    axt: Axes | None = None,
    loc: str = "best",
    ncol: int = 1,
    title: str | None = None,
    bbox_xy: tuple[float] | None = None,
    font_size: int = 10,
    borderpad: float = 0.3,
    handlelength: float = 1.5,
    masked_values: bool = False,
):
    hnd_ax, lab_ax = ax.get_legend_handles_labels()
    if axt is not None:
        hnd_axt, lab_axt = axt.get_legend_handles_labels()
    else:
        hnd_axt = []
        lab_axt = []
    ax.legend(
        hnd_ax + hnd_axt,
        lab_ax + lab_axt,
        loc=loc,
        ncol=ncol,
        title=title,
        bbox_to_anchor=(bbox_xy if bbox_xy is not None else None),
        fontsize=font_size,
        borderpad=borderpad,
        handlelength=handlelength,
    )
    if masked_values:
        for handle in ax.legend().legendHandles:
            handle.set_alpha(1)  # Set alpha of each legend handle to fully opaque


def _mask_insignificant_data_in_ax(ax, alpha: float = 0.3) -> None:
    axbars = [b for b in ax.patches if isinstance(b, mpatches.Rectangle)]

    if not axbars:
        return

    df_ave, df_std = _extract_ave_std_from_ax(ax)
    ave_values = df_ave.T.to_numpy().ravel().tolist()
    std_values = df_std.T.to_numpy().ravel().tolist()

    # Iterate over axbars and their corresponding error axbars
    for i, axbar in enumerate(axbars):
        std = std_values[i]
        ave = ave_values[i]
        if std > ave:
            axbar.set_alpha(alpha)
        else:
            axbar.set_alpha(1.0)


def _annotate_letters_to_ax(ax, letter: str, xy: tuple[float], font_size: int) -> None:
    """
    Annotate the subplots with letters.
    """
    ax.annotate(
        f"({letter})",
        xycoords="axes fraction",
        xy=(0, 0),
        xytext=xy,
        size=font_size,
        weight="bold",
    )


def _broadcast_value_prop(
    prop: list | str | float | int | bool, prop_name: str, number_of_axis: int
) -> list:
    """
    Broadcast a single value property to a list applicable to all subplots.

    :param prop: The property to broadcast.
    :type prop: Union[list, str, float, int, bool]
    :param prop_name: The name of the property, used in error messages.
    :type prop_name: str
    :return: A list of the property values broadcasted to match the number of subplots.
    :rtype: list
    """
    if prop is None:
        prop = [None] * number_of_axis
    elif isinstance(prop, (list, tuple)):
        if len(prop) != number_of_axis:
            raise ValueError(
                f"The size of the property '{prop_name}' does not match the number of axes."
            )
    elif isinstance(prop, (str, float, int, bool)):
        prop = [prop] * number_of_axis
    return prop


def _broadcast_list_prop(prop: list | None, prop_name: str, number_of_axis: int):
    """_summary_

    :param prop: _description_
    :type prop: list | None
    :param prop_name: The name of the property for error messages.
    :type prop_name: str
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """
    if prop is None:
        prop = [None] * number_of_axis
    # Check if prop is a list of lists and has the correct length
    elif all(isinstance(item, (list, tuple)) for item in prop) and len(prop) != number_of_axis:
        raise ValueError(
            f"The size of the property '{prop_name}' does not match the number of axes."
        )
    elif isinstance(prop, (list, tuple)) and all(
        isinstance(item, (int, float, str)) for item in prop
    ):
        prop = [prop] * number_of_axis
    return prop


def _extract_ave_std_from_ax(ax):
    axbars = [b for b in ax.patches if isinstance(b, mpatches.Rectangle)]
    ave_values = [axbar.get_height() for axbar in axbars]
    std_values = [0] * len(axbars)  # Initialize a list of zeros for standard deviations

    # Collect all LineCollections
    line_collections = [col for col in ax.collections if isinstance(col, LineCollection)]

    # Assuming error bars are vertically oriented, extract standard deviations
    index = 0  # Start index for assigning std values from segments
    for lc in line_collections:
        segments = lc.get_segments()
        for seg in segments:
            std = (seg[1][1] - seg[0][1]) / 2
            if index < len(std_values):  # Ensure we do not go out of index range
                std_values[index] = std
                index += 1

    df_ave = pd.DataFrame([ave_values], columns=[f"Bar {i+1}" for i in range(len(axbars))])
    df_std = pd.DataFrame([std_values], columns=df_ave.columns)

    return df_ave, df_std


def _rotate_x_labels_ax(ax, rotation: float | int) -> None:
    """
    Rotate the labels on the x-axis.
    avoids th
    """
    # Directly set the rotation for existing tick labels
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        if rotation != 0:
            label.set_ha("right")
            label.set_rotation_mode("anchor")


def _annotate_outliers_to_ax(ax, decimal_places=2) -> None:

    axbars = [b for b in ax.patches if isinstance(b, mpatches.Rectangle)]
    if not axbars:
        return

    # Set dx and dy for text positioning adjustments
    df_ave, df_std = _extract_ave_std_from_ax(ax)
    y_lim = ax.get_ylim()
    dx = 0.15 * len(df_ave.index)
    dy = 0.04
    tform = blended_transform_factory(ax.transData, ax.transAxes)
    dfao = pd.DataFrame(columns=["H/L", "xpos", "ypos", "ave", "std", "text"])
    # Flatten and assign average and standard deviation values
    dfao["ave"] = df_ave.T.to_numpy().ravel().tolist()
    dfao["std"] = (
        df_std.T.to_numpy().ravel().tolist() if not df_std.empty else np.zeros(dfao["ave"].size)
    )

    # Determine x positions of axbars
    try:
        dfao["xpos"] = [p.get_x() + p.get_width() / 2 for p in ax.patches]
    except ValueError:  # Correct for possible duplicates due to masking
        dfao["xpos"] = [p.get_x() + p.get_width() / 2 for p in ax.patches[: len(ax.patches) // 2]]

    # Drop rows outside of y limits
    dfao = dfao[(dfao["ave"] < y_lim[0]) | (dfao["ave"] > y_lim[1])]
    # Loop through bars to set text and H/L values
    for ao in dfao.index:
        ave = dfao.at[ao, "ave"]
        std = dfao.at[ao, "std"]
        if ave == float("inf"):
            text = "inf"
            hl = "H"
        elif ave == float("-inf"):
            text = "-inf"
            hl = "L"
        elif ave > y_lim[1]:
            hl = "H"
            text = f"{ave:.{decimal_places}f}"
            if std != 0 and not np.isnan(std):
                text += rf"$\pm${std:.{decimal_places}f}"
        elif ave < y_lim[0]:
            hl = "L"
            text = f"{ave:.{decimal_places}f}"
            if std != 0:
                text += rf"$\pm${std:.{decimal_places}f}"
        else:
            print("Something is wrong", ave)
        dfao.loc[ao, "text"] = text
        dfao.loc[ao, "H/L"] = hl

    for hl, ypos, dy in zip(["L", "H"], [0.02, 0.98], [0.04, -0.04]):
        dfao1 = dfao[dfao["H/L"] == hl]  # pylint: disable=unsubscriptable-object
        dfao1["ypos"] = ypos
        if not dfao1.empty:
            dfao1 = dfao1.sort_values("xpos", ascending=True)
            dfao1["diffx"] = np.diff(dfao1["xpos"].values, prepend=dfao1["xpos"].values[0]) < dx
            dfao1.reset_index(inplace=True)

            for i in dfao1.index.tolist()[1:]:
                dfao1.loc[i, "ypos"] = ypos
                for e in range(i, 0, -1):
                    if dfao1.loc[e, "diffx"]:
                        dfao1.loc[e, "ypos"] += dy
                    else:
                        break
            for ao in dfao1.index.tolist():
                ax.annotate(
                    dfao1.loc[ao, "text"],
                    xy=(dfao1.loc[ao, "xpos"], 0),
                    xycoords=tform,
                    textcoords=tform,
                    xytext=(dfao1.loc[ao, "xpos"], dfao1.loc[ao, "ypos"]),
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "square,pad=0",
                        "edgecolor": None,
                        "facecolor": "white",
                        "alpha": 0.7,
                    },
                )


def _apply_hatch_patterns_to_ax(ax) -> None:
    """
    Apply hatch patterns to bars in the bar plots of each subplot.

    This method iterates over all subplots and applies predefined hatch patterns to each bar,
    enhancing the visual distinction between axbars, especially in black and white printouts.
    """
    # Check if the plot is a bar plot
    axbars = [b for b in ax.patches if isinstance(b, mpatches.Rectangle)]
    # If there are no axbars, return immediately
    if not axbars:
        return
    num_groups = len(ax.get_xticks(minor=False))
    # Determine the number of axbars in each group
    bars_in_group = len(axbars) // num_groups
    patterns = hatches[:bars_in_group]  # set hatch patterns in correct order
    plot_hatches_list = []  # list for hatches in the order of the axbars
    for h in patterns:  # loop over patterns to create bar-ordered hatches
        for _ in range(int(len(axbars) / len(patterns))):
            plot_hatches_list.append(h)
    # loop over axbars and hatches to set hatches in correct order
    for b, hatch in zip(axbars, plot_hatches_list):
        b.set_hatch(hatch)
        b.set_edgecolor("k")
