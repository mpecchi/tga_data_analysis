import pytest
from tga_data_analysis.myfigure import MyFigure
import matplotlib.pyplot as plt


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
    fig = MyFigure(rows=2, cols=2)  # Assuming a 2x2 grid
    # Broadcasting a single value
    result = fig._broadcast_value_prop("test", "prop_name")
    assert result == ["test", "test", "test", "test"]

    # Broadcasting a list of values
    result = fig._broadcast_value_prop(["a", "b", "c", "d"], "prop_name")
    assert result == ["a", "b", "c", "d"]

    # Broadcasting with incorrect list size
    with pytest.raises(ValueError):
        fig._broadcast_value_prop(["a", "b"], "prop_name")


def test_broadcast_list_prop():
    fig = MyFigure(rows=2, cols=1)  # Assuming a 2x1 grid
    # Broadcasting a list of lists
    result = fig._broadcast_list_prop([[1, 2], [3, 4]], "prop_name")
    assert result == [[1, 2], [3, 4]]

    # Broadcasting a list with incorrect inner list size
    with pytest.raises(ValueError):
        fig._broadcast_list_prop([[1, 2], [3]], "prop_name")


def test_broadcast_value_prop_single_axis():
    fig = MyFigure(rows=1, cols=1)  # A single subplot
    # Broadcasting a single value
    result = fig._broadcast_value_prop("test", "prop_name")
    assert result == ["test"]

    # Broadcasting a list of values (which should be just one value for a single subplot)
    result = fig._broadcast_value_prop(["a"], "prop_name")
    assert result == ["a"]

    # Broadcasting with incorrect list size should raise an error even for a single subplot
    with pytest.raises(ValueError):
        fig._broadcast_value_prop(["a", "b"], "prop_name")


def test_broadcast_list_prop_single_axis():
    fig = MyFigure(rows=1, cols=1)  # A single subplot
    # Broadcasting a list of lists (which should just be one list for a single subplot)
    result = fig._broadcast_list_prop([[1, 2]], "prop_name")
    assert result == [[1, 2]]

    # Broadcasting a list with incorrect inner list size should raise an error
    with pytest.raises(ValueError):
        fig._broadcast_list_prop([[1, 2], [3]], "prop_name")


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
