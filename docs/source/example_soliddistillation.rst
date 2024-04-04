Solid Distillation Analysis Example
===================================

In this example, we demonstrate how to use `tga_data_analysis` for solid distillation analysis, a process that helps understand the thermal decomposition behavior of solid materials.

Setting Up the Project and Samples
----------------------------------

First, we initialize a project and create two samples representing different conditions for solid distillation analysis:

.. code-block:: python

    import pathlib as plib
    from tga_data_analysis.tga import Project, Sample

    folder_path = plib.Path(__file__).resolve().parent / "soliddistillation"

    proj_soliddist = Project(folder_path, name="test", temp_unit="K")

    sda = Sample(
        project=proj_soliddist,
        name="sda",
        filenames=["SDa_1", "SDa_2", "SDa_3"],
        time_moist=38,
        time_vm=None,
    )
    sdb = Sample(
        project=proj_soliddist,
        name="sdb",
        filenames=["SDb_1", "SDb_2", "SDb_3"],
        time_moist=38,
        time_vm=None,
    )

Conducting Solid Distillation Analysis
--------------------------------------

We generate reports and plots for each sample to analyze their solid distillation behavior:

.. code-block:: python

    repa = sda.report("soliddist")
    repb = sdb.report("soliddist")

    mf = sda.plot_soliddist()
    mf = sdb.plot_soliddist()

Additionally, we create a combined report and a comparative plot to visualize the differences in distillation behavior between the samples:

.. code-block:: python

    rep = proj_soliddist.multireport(report_type="soliddist")
    mf = proj_soliddist.plot_multi_soliddist(labels=["sample 1", "sample 2", "sample 3"])

This example outlines the process of conducting solid distillation analysis, showcasing how to interpret the thermal decomposition characteristics of different solid samples using `tga_data_analysis`.
