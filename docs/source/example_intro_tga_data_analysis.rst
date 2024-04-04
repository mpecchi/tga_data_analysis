Introduction to TGA Data Analysis
=================================

This example provides a basic introduction to using the `tga_data_analysis` package. We will create two `Project` instances with different configurations and a `Sample` instance to analyze thermogravimetric analysis (TGA) data.

Setting Up the Environment
--------------------------

First, we import the necessary modules and define the directory where our TGA data is located:

.. code-block:: python

    import pathlib as plib
    from tga_data_analysis.tga import Project, Sample

    test_dir = plib.Path(__file__).resolve().parent / "project"

Creating Projects
-----------------

We create two `Project` instances with different settings to demonstrate the flexibility of the package:

1. **Default Project**: This project uses Celsius for temperature units and sets various parameters for the analysis.

   .. code-block:: python

       proj_default = Project(
           test_dir,
           name="default",
           temp_unit="C",
           plot_font="Dejavu Sans",
           dtg_basis="temperature",
           resolution_sec_deg_dtg=5,
           dtg_window_filter=101,
           plot_grid=False,
           temp_initial_celsius=40,
           temp_lim_dtg_celsius=None,
       )

2. **Modified Project**: This project uses Kelvin for temperature units and different parameters to show how they impact the analysis.

   .. code-block:: python

       proj_mod = Project(
           test_dir,
           name="mod",
           temp_unit="K",
           plot_font="Times New Roman",
           dtg_basis="time",
           resolution_sec_deg_dtg=2,  # Illustrative choice to show parameter impact
           dtg_window_filter=201,     # Illustrative choice to show parameter impact
           plot_grid=True,
           temp_initial_celsius=50,
           temp_lim_dtg_celsius=(150, 750),
       )

Analyzing Samples
-----------------

For each project, we create a `Sample` instance and generate TG/DTG plots:

.. code-block:: python

    cell = Sample(
        project=proj_default,
        name="cell",
        filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
        time_moist=38,
        time_vm=None,
    )
    mf = cell.plot_tg_dtg()

    cell = Sample(
        project=proj_mod,
        name="cell",
        filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
        time_moist=38,
        time_vm=None,
    )
    mf = cell.plot_tg_dtg()

This code demonstrates the basic workflow of defining projects and samples, highlighting the package's capability to analyze and visualize TGA data effectively.
