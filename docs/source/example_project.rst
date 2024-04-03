Example Project Usage
=====================

This example demonstrates how to use the ``Project`` and ``Sample`` classes within the ``tga_data_analysis.tga`` module to analyze thermogravimetric analysis (TGA) data. The code snippet below shows the instantiation of project and sample objects, the configuration of various parameters, and how to generate thermogravimetric (TG) and derivative thermogravimetric (DTG) plots.

.. code-block:: python

    import pathlib as plib
    from tga_data_analysis.tga import Project, Sample

    # Assume test_dir points to the directory where the TGA data files are located
    # This directory is structured as expected by the Project and Sample classes
    test_dir = plib.Path(__file__).resolve().parent / "project"

    # Creating a Project instance with default settings
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

    # Creating a Sample instance associated with the 'default' project
    # The Sample instance represents a specific TGA experiment with its data files
    cell = Sample(
        project=proj_default,
        name="cell",
        filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
        time_moist=38,
        time_vm=None,
    )

    # Generating a TG/DTG plot for the 'cell' sample
    mf = cell.plot_tg_dtg()

    # Creating another Project instance with modified settings to demonstrate flexibility
    proj_mod = Project(
        test_dir,
        name="mod",
        temp_unit="K",
        plot_font="Times New Roman",
        dtg_basis="time",
        resolution_sec_deg_dtg=2,  # Example showing the impact of this parameter
        dtg_window_filter=201,  # Example showing the impact of this parameter
        plot_grid=True,
        temp_initial_celsius=50,
        temp_lim_dtg_celsius=(150, 750),
    )

    # Creating another Sample instance under the 'mod' project
    cell = Sample(
        project=proj_mod,
        name="cell",
        filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
        time_moist=38,
        time_vm=None,
    )

    # Generating another TG/DTG plot to observe the differences based on project settings
    mf = cell.plot_tg_dtg()
