Oxidation Analysis Example
==========================

This example illustrates how to use `tga_data_analysis` for oxidation analysis. Unlike combustion analysis, which occurs at a faster rate, oxidation analysis focuses on slower processes, as indicated by the proximate curve.

Setting Up the Project and Samples
----------------------------------

We initiate a project and define two samples under different conditions to analyze their oxidation behavior:

.. code-block:: python

    import pathlib as plib
    from tga_data_analysis.tga import Project, Sample

    folder_path = plib.Path(__file__).resolve().parent / "oxidation_analysis"

    proj = Project(folder_path=folder_path, name="test", temp_unit="K")
    cell_ox5 = Sample(
        project=proj,
        name="cell_ox5",
        filenames=["CLSOx5_1", "CLSOx5_2", "CLSOx5_3"],
        time_moist=38,
        heating_rate_deg_min=5,
    )
    cell_ox10 = Sample(
        project=proj,
        name="cell_ox10",
        load_skiprows=8,
        filenames=["CLSOx10_2", "CLSOx10_3", "CLSOx10_4"],
        time_moist=38,
        heating_rate_deg_min=10,
    )

Conducting the Analysis
-----------------------

The samples are analyzed for their oxidation properties. The analysis generates reports and plots to visualize the TG/DTG curves:

.. code-block:: python

    for sample in proj.samples.values():
        for report_type in ["oxidation", "oxidation_extended"]:
            sample.report(report_type)
            mf = sample.plot_tg_dtg()

    # Generating a multi-sample oxidation report
    rep = proj.multireport(report_type="oxidation")

    # Plotting multi-sample DTG curves
    mf = proj.plot_multi_dtg()

This sequence of steps demonstrates how to conduct a detailed oxidation analysis using `tga_data_analysis`, offering insights into the thermal behavior of materials under oxidative conditions.
