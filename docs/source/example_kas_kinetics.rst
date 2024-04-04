KAS Kinetics Analysis Example
=============================

This example demonstrates the use of the `tga_data_analysis` package to perform KAS kinetics analysis, a method for determining the activation energy of a material from TGA data.

Setting Up the Project and Samples
----------------------------------

First, we initialize a project and define several samples at different heating rates to prepare for KAS analysis:

.. code-block:: python

    import pathlib as plib
    from tga_data_analysis.tga import Project, Sample
    from tga_data_analysis.kas_kinetics import KasSample, plot_multi_activation_energy

    folder_path = plib.Path(__file__).resolve().parent / "kas_kinetics"
    proj = Project(folder_path=folder_path, name="test", temp_unit="K")

    # Defining samples for cellulose
    cell_ox5 = Sample(
        project=proj,
        name="cell_ox5",
        filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
        heating_rate_deg_min=5,
    )
    # Additional samples for cellulose at different heating rates...

    # Defining samples for a primary component at different heating rates...
    pc_ox10 = Sample(
        project=proj,
        name="pc_ox10",
        filenames=["PCOx10_1"],
        heating_rate_deg_min=10,
    )
    # Additional samples for the primary component...

Performing KAS Analysis
-----------------------

We perform KAS analysis on the defined samples, generating isoline and activation energy plots:

.. code-block:: python

    # Cellulose samples
    cell = KasSample(proj, samples=[cell_ox5, cell_ox10, cell_ox50, cell_ox100], name="cellulose")
    mf = cell.plot_isolines(legend_bbox_xy=(1, 1))
    mf = cell.plot_activation_energy(legend_bbox_xy=(1, 1))

    # Primary component samples
    pc = KasSample(proj, samples=[pc_ox10, pc_ox50, pc_ox100], name="primary")
    mf = pc.plot_isolines(legend_bbox_xy=(1, 1))
    mf = pc.plot_activation_energy(legend_bbox_xy=(1, 1))

    # Comparative activation energy plot
    mf = plot_multi_activation_energy([cell, pc])

This example provides a comprehensive guide to performing KAS kinetics analysis with `tga_data_analysis`, showcasing how to extract and compare the activation energy of different materials based on their TGA data.
