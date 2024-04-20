# %%
from __future__ import annotations
import matplotlib.pyplot as plt
import pytest
from tga_data_analysis.myfigure import MyFigure


def test_myfigure_initialization_with_defaults():
    # Test initialization with default values
    fig = MyFigure()
    assert isinstance(fig.broad_props, dict)
    assert isinstance(fig.kwargs, dict)
    assert "rows" in fig.kwargs and fig.kwargs["rows"] == 1  # Default value assumed to be 1
    assert "cols" in fig.kwargs and fig.kwargs["cols"] == 1  # Default value assumed to be 1
    assert isinstance(fig.fig, plt.Figure)  # Check if fig is an instance of plt.Figure
    assert len(fig.axs) == 1  # There should be one Axes object in the list


def test_myfigure_initialization_with_custom_values():
    # Test initialization with custom values
    custom_kwargs = {"rows": 2, "cols": 3, "width": 10.0}
    fig = MyFigure(**custom_kwargs)
    for key, value in custom_kwargs.items():
        assert key in fig.kwargs and fig.kwargs[key] == value


def test_broadcast_value_prop():
    fig = MyFigure(rows=2, cols=2, x_lab="test", y_lab=["a", "b", "c", "d"])  # Assuming a 2x2 grid
    # Broadcasting a single value
    result = fig.broad_props["x_lab"]
    assert result == ["test", "test", "test", "test"]

    # Broadcasting a list of values
    result = fig.broad_props["y_lab"]
    assert result == ["a", "b", "c", "d"]

    # Broadcasting with incorrect list size
    with pytest.raises(ValueError):
        fig = MyFigure(rows=2, cols=2, y_lab=["a", "b"])


def test_broadcast_list_prop():
    fig = MyFigure(rows=2, cols=1, x_lim=(0, 1), y_lim=[(0, 1), (2, 4)])  # Assuming a 2x1 grid
    # Broadcasting a list of lists
    result = fig.broad_props["x_lim"]
    assert result == [(0, 1), (0, 1)]

    result = fig.broad_props["y_lim"]
    assert result == [(0, 1), (2, 4)]

    # Broadcasting a list with incorrect inner list size
    with pytest.raises(ValueError):
        fig = MyFigure(rows=2, cols=1, y_lim=[(0, 1), (2, 4), (3, 5)])


def test_create_figure_single_axis():
    fig = MyFigure(rows=1, cols=1)
    fig.create_figure()
    assert isinstance(fig.fig, plt.Figure)
    assert len(fig.axs) == 1  # Should have one Axes object in the list
    assert isinstance(fig.axs[0], plt.Axes)


def test_create_figure_multiple_axes():
    rows, cols = 2, 3
    fig = MyFigure(rows=rows, cols=cols)
    fig.create_figure()
    assert isinstance(fig.fig, plt.Figure)
    assert len(fig.axs) == rows * cols  # Number of Axes should match rows * cols
    for ax in fig.axs:
        assert isinstance(ax, plt.Axes)


def test_invalid_rows_cols():
    # Testing invalid row and column inputs
    with pytest.raises(ValueError):
        MyFigure(rows=-1, cols=1)
    with pytest.raises(ValueError):
        MyFigure(rows=1, cols=0)


def test_invalid_kwargs():
    # Testing with an invalid keyword argument
    with pytest.raises(ValueError):
        MyFigure(invalid_arg=123)


# %%
