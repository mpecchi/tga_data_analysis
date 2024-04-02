from __future__ import annotations
import string
import pathlib as plib
from typing import Any, Dict, Literal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns


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

htchs: list[str] = [
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

    @staticmethod
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

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize a MyFigure object with optional configuration.

        :param kwargs: Configuration options as keyword arguments.
        :type kwargs: Any
        """
        self.broad_props: dict[str, list] = {}  # broadcasted properties for each axis
        self.kwargs = self.default_kwargs()
        self.kwargs.update(kwargs)  # Override defaults with any kwargs provided
        self.process_kwargs()

        sns.set_palette(self.kwargs["color_palette"], self.kwargs["color_palette_n_colors"])
        sns.set_style(self.kwargs["sns_style"], {"font.family": self.kwargs["text_font"]})

        self.create_figure()

        self.update_axes_single_props()

        self.update_axes_list_props()

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
            "annotate_lttrs": None,
            "annotate_lttrs_xy": None,
            "grid": None,
            "color_palette": "deep",
            "color_palette_n_colors": None,
            "text_font": "Dejavu Sans",
            "sns_style": "ticks",
        }
        return defaults

    def process_kwargs(self) -> None:
        """
        Process and validate the provided keyword arguments, ensuring they are compatible with the figure settings.

        Raises a ValueError if an invalid keyword argument is provided or if required arguments are missing.
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
                raise ValueError("Filename must be a str.")
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

    def create_figure(self) -> MyFigure:
        """
        Creates the figure and its axes.

        :return: MyFigure
        :rtype: MyFigure
        """
        self.fig: Figure
        self.axs: Axes
        self.axts: Axes | None
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

    def save_figure(
        self,
        filename: str | None = None,
        out_path: plib.Path | None = None,
        tight_layout: bool = True,
        save_as_png: bool = True,
        save_as_pdf: bool = False,
        save_as_svg: bool = False,
        save_as_eps: bool = False,
        png_transparency: bool = False,
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
        self.update_axes_single_props()

        self.update_axes_list_props()

        self.apply_hatch_patterns()

        self.add_legend()

        self.rotate_x_labels()

        try:
            self.fig.align_labels()  # align labels of subplots, needed only for multi plot
        except AttributeError:
            print("align_labels not performed")
        self.annotate_letters()
        # Saving the figure
        formats = {
            "png": save_as_png,
            "pdf": save_as_pdf,
            "svg": save_as_svg,
            "eps": save_as_eps,
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
                    dpi=300,
                    transparent=png_transparency,
                    bbox_inches="tight" if tight_layout else None,
                )

    def add_legend(self) -> None:
        """
        Add a legend to the figure.
        """
        for sprop in ["legend", "legend_loc", "legend_ncols", "legend_title"]:
            self.broad_props[sprop] = self._broadcast_value_prop(self.kwargs[sprop], sprop)
        for lprop in ["legend_bbox_xy"]:
            self.broad_props[lprop] = self._broadcast_list_prop(self.kwargs[lprop], lprop)

        if self.kwargs["twinx"] is None:

            for i, ax in enumerate(self.axs):
                if self.broad_props["legend"][i]:
                    ax.legend(
                        loc=self.broad_props["legend_loc"][i],
                        ncol=self.broad_props["legend_ncols"][i],
                        title=self.broad_props["legend_title"][i],
                        bbox_to_anchor=(
                            self.broad_props["legend_bbox_xy"][i]
                            if self.broad_props["legend_bbox_xy"][i] is not None
                            else None
                        ),
                    )

        else:
            for i, (ax, axt) in enumerate(zip(self.axs, self.axts)):
                if self.broad_props["legend"][i]:
                    hnd_ax, lab_ax = ax.get_legend_handles_labels()
                    hnd_axt, lab_axt = axt.get_legend_handles_labels()
                    ax.legend(
                        hnd_ax + hnd_axt,
                        lab_ax + lab_axt,
                        loc=self.broad_props["legend_loc"][i],
                        ncol=self.broad_props["legend_ncols"][i],
                        title=self.broad_props["legend_title"][i],
                        bbox_to_anchor=(
                            self.broad_props["legend_bbox_xy"][i]
                            if self.broad_props["legend_bbox_xy"][i] is not None
                            else None
                        ),
                    )

    def annotate_letters(self) -> None:
        """
        Annotate the subplots with letters.
        """
        if (
            self.kwargs["annotate_lttrs_xy"] is not None
            and isinstance(self.kwargs["annotate_lttrs_xy"], (list, tuple))
            and len(self.kwargs["annotate_lttrs_xy"]) >= 2
        ):
            xylttrs: list | tuple = self.kwargs["annotate_lttrs_xy"]
            x_lttrs = xylttrs[0]  # pylint: disable=unsubscriptable-object
            y_lttrs = xylttrs[1]  # pylint: disable=unsubscriptable-object
        else:
            x_lttrs = -0.15
            y_lttrs = -0.15
        if self.kwargs["annotate_lttrs"] is not None:
            if isinstance(self.kwargs["annotate_lttrs"], str):
                letters_list = [self.kwargs["annotate_lttrs"]]
            elif isinstance(self.kwargs["annotate_lttrs"], (list, tuple)):
                letters_list = self.kwargs["annotate_lttrs"]
            for i, ax in enumerate(self.axs):
                ax.annotate(
                    f"({letters_list[i]})",
                    xycoords="axes fraction",
                    xy=(0, 0),
                    xytext=(x_lttrs, y_lttrs),
                    size="large",
                    weight="bold",
                )

    def create_inset(
        self,
        ax: Axes,
        ins_x_loc: tuple[float],
        ins_y_loc: tuple[float],
        ins_x_lim: tuple[float] | None = None,
        ins_y_lim: tuple[float] | None = None,
    ) -> Axes:
        """
        Create an inset plot within an existing axis.

        :param ax: The parent axis.
        :type ax: Axes
        :param ins_x_loc: X location for the inset.
        :type ins_x_loc: tuple[float, float]
        :param ins_y_loc: Y location for the inset.
        :type ins_y_loc: tuple[float, float]
        :param ins_x_lim: X limits for the inset.
        :type ins_x_lim: tuple[float, float] | None
        :param ins_y_lim: Y limits for the inset.
        :type ins_y_lim: tuple[float, float] | None
        :return: The inset axes.
        :rtype: Axes
        """
        wdt = ins_x_loc[1] - ins_x_loc[0]
        hgt = ins_y_loc[1] - ins_y_loc[0]
        inset = ax.inset_axes([ins_x_loc[0], ins_y_loc[0], wdt, hgt])
        if ins_x_lim is not None:
            inset.set_xlim(MyFigure._adjust_lims(ins_x_lim))
        if ins_y_lim is not None:
            inset.set_ylim(MyFigure._adjust_lims(ins_y_lim))
        return inset

    def update_axes_single_props(self):
        """
        Update properties that are applied to each axis individually.
        """
        for sprop in [
            "x_lab",
            "y_lab",
            "yt_lab",
            "grid",
        ]:
            self.broad_props[sprop] = self._broadcast_value_prop(self.kwargs[sprop], sprop)

        # Update each axis with the respective properties
        for i, ax in enumerate(self.axs):
            ax.set_xlabel(self.broad_props["x_lab"][i])
            ax.set_ylabel(self.broad_props["y_lab"][i])
            if self.broad_props["grid"][i] is not None:
                ax.grid(self.broad_props["grid"][i])
        if self.kwargs["twinx"]:
            for i, axt in enumerate(self.axts):
                axt.set_ylabel(self.broad_props["yt_lab"][i])

    def update_axes_list_props(self):
        """
        Update list properties for the axes.
        """
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
        ]:
            self.broad_props[lprop] = self._broadcast_list_prop(self.kwargs[lprop], lprop)

        # Update each axis with the respective properties
        for i, ax in enumerate(self.axs):
            if self.broad_props["x_lim"][i] is not None:
                ax.set_xlim(MyFigure._adjust_lims(self.broad_props["x_lim"][i]))
            if self.broad_props["y_lim"][i] is not None:
                ax.set_ylim(MyFigure._adjust_lims(self.broad_props["y_lim"][i]))
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
                if self.broad_props["yt_lim"][i] is not None:
                    axt.set_ylim(MyFigure._adjust_lims(self.broad_props["yt_lim"][i]))
                if self.broad_props["yt_ticks"][i] is not None:
                    axt.set_yticks(self.broad_props["yt_ticks"][i])
                if self.broad_props["yt_ticklabels"][i] is not None:
                    axt.set_yticklabels(self.broad_props["yt_ticklabels"][i])

    def rotate_x_labels(self):
        """
        Rotate the labels on the x-axis.
        """
        self.broad_props["x_ticklabels_rotation"] = self._broadcast_value_prop(
            self.kwargs["x_ticklabels_rotation"], "x_ticklabels_rotation"
        )

        # Update each axis with the respective properties
        for i, ax in enumerate(self.axs):
            rotation = self.broad_props["x_ticklabels_rotation"][i]

            # Directly set the rotation for existing tick labels
            for label in ax.get_xticklabels():
                label.set_rotation(rotation)
                if rotation != 0:
                    label.set_ha("right")
                    label.set_rotation_mode("anchor")

    def apply_hatch_patterns(self):
        """
        Apply hatch patterns to bars in the bar plots of each subplot.

        This method iterates over all subplots and applies predefined hatch patterns to each bar,
        enhancing the visual distinction between bars, especially in black and white printouts.
        """
        for ax in self.axs:
            # Check if the plot is a bar plot
            bars = [bar for bar in ax.patches if isinstance(bar, mpatches.Rectangle)]
            # If there are no bars, return immediately
            if not bars:
                return
            num_groups = len(ax.get_xticks(minor=False))
            # Determine the number of bars in each group
            bars_in_group = len(bars) // num_groups
            patterns = htchs[:bars_in_group]  # set hatch patterns in correct order
            hatches = []  # list for hatches in the order of the bars
            for h in patterns:  # loop over patterns to create bar-ordered hatches
                for i in range(int(len(bars) / len(patterns))):
                    hatches.append(h)
            # loop over bars and hatches to set hatches in correct order
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)

    def _broadcast_value_prop(self, prop: list | str | float | int | bool, prop_name: str) -> list:
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
            return [None] * self.n_axs
        if isinstance(prop, (list, tuple)):
            if len(prop) == self.n_axs:
                return prop
            else:
                raise ValueError(
                    f"The size of the property '{prop_name}' does not match the number of axes."
                )
        if isinstance(prop, (str, float, int, bool)):
            return [prop] * self.n_axs

    def _broadcast_list_prop(self, prop: list | None, prop_name: str):
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
            return [None] * self.n_axs

        # Check if prop is a list of lists and has the correct length
        if all(isinstance(item, (list, tuple)) for item in prop) and len(prop) == self.n_axs:
            # Ensure all inner lists have the same size
            if not all(len(item) == len(prop[0]) for item in prop):
                raise ValueError(f"All inner lists in '{prop_name}' must have the same size.")
            return prop
        elif isinstance(prop, (list, tuple)) and all(
            isinstance(item, (int, float, str)) for item in prop
        ):
            return [prop] * self.n_axs
        else:
            raise ValueError(f"The structure of '{prop_name}' does not match the expected input.")


if __name__ == "__main__":
    f = MyFigure(
        filename="my_plot",
        out_path=plib.Path(r"C:\Users\mp933\Desktop\New folder"),
        rows=4,
        cols=1,
        width=6,
        height=12,
        twinx=True,
        x_lab=["aaa", "qqq", "aa", "qq"],
        y_lab="bbb",
        yt_lab="ccc",
        x_lim=[0, 1],
        y_lim=[0, 1],
        yt_lim=[[0, 1], [0, 0.5], [0, 1], [0, 0.5]],
        x_ticks=[[0, 0.5, 1], [0, 0.5, 2], [0, 1], [0, 0.5]],
        # x_ticklabels=["a", "c", "d"],
        grid=True,
        annotate_lttrs=["a", "b", "a", "b"],
        annotate_lttrs_xy=[-0.11, -0.15],
        x_ticklabels_rotation=0,
    )

    f.axs[0].plot([0, 1], [0, 3], label="a")
    f.axts[0].plot([0, 2], [0, 4], label="b")
    f.axts[0].plot([0, 2], [0, 5], label="ccc")
    f.axs[1].plot([0, 1], [0, 3], label="aaa")
    ins = f.create_inset(f.axs[0], [0.6, 0.8], [0.4, 0.6], [0, 0.2], [0, 0.2])
    ins.plot([0, 1], [0, 3], label="a")
    f.save_figure()
