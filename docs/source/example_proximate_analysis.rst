Proximate Analysis Example
==========================

This example demonstrates how to perform a proximate analysis using `tga_data_analysis`. Proximate analysis is a valuable method in thermal analysis, providing insights into the composition of materials.

Setting Up the Project and Samples
----------------------------------

First, we set up the project and define two samples, `sru` and `misc`, which will be analyzed:

.. code-block:: python

    import pathlib as plib
    from tga_data_analysis.tga import Project, Sample

    test_dir = plib.Path(__file__).resolve().parent / "proximate_analysis"

    proj = Project(test_dir, name="test", temp_unit="K")
    sru = Sample(
        project=proj, name="sru", filenames=["SRU_1", "SRU_2", "SRU_3"], time_moist=38, time_vm=167
    )
    misc = Sample(
        project=proj, name="misc", filenames=["MIS_1", "MIS_2", "MIS_3"], time_moist=38, time_vm=147
    )

Performing the Analysis
-----------------------

The samples are analyzed to generate TG/DTG plots and a multireport for proximate analysis:

.. code-block:: python

    proximate_samples = [sru, misc]
    for sample in proximate_samples:
        _ = sample.plot_tg_dtg()

    # Generating a multi-sample report
    rep = proj.multireport(
        samples=proximate_samples,
        report_type="proximate",
    )

    # Plotting the multi-sample report
    rep = proj.plot_multireport(
        "rep_",
        samples=proximate_samples,
        report_type="proximate",
        height=4,
        width=4.5,
        y_lim=(0, 100),
    )

    # Generating DTG comparison plots
    rep = proj.plot_multi_dtg(
        "rep",
        samples=proximate_samples,
        height=4,
        width=4.5,
    )

This example showcases how `tga_data_analysis` can be used to conduct a proximate analysis, comparing multiple samples and visualizing their thermal decomposition characteristics.
