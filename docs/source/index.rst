.. tga_data_analysis documentation master file, created by
   sphinx-quickstart on Mon Apr  1 21:47:38 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction to Automated TGA Data Analysis
===========================================

The `tga_data_analysis` tool is designed to streamline and automate the analysis of thermogravimetric analysis (TGA) data, helping TGA users saving time and avoiding human error   . This tool is developed with the Python ecosystem in mind, ensuring seamless integration with various data analysis workflows.

**Key Features:**

- **Proximate Analysis**: Determines the moisture, volatile matter, and ash content from TGA data.
- **Oxidation Analysis**: Analyzes the oxidation behavior of materials.
- **Solid-Distillation Analysis**: Studies the thermal decomposition and distillation characteristics of solids.
- **KAS Kinetic Analysis**: Applies the Kissinger-Akahira-Sunose method to determine kinetic parameters.
- **Peak Deconvolution**: Resolves overlapping thermal decomposition events.

**Conceptual Framework:**

- **Project**: Represents a folder containing all TGA data for a specific campaign, adhering to uniform requirements such as temperature units and plot styles.
- **Sample**: A collection of replicate runs of the same material under identical analysis conditions, providing mean values and standard deviations for each result.
- **File**: An individual TGA run.
- **Report**: Can be generated for a single sample (exploring each replicate) or multiple samples (comparing averages and standard deviations).

This tool is crafted to enhance efficiency and provide insightful analyses, assisting researchers and professionals in making informed decisions based on their TGA data.


.. toctree::
   :maxdepth: 4
   :caption: Contents:

   installation

   getting_started

   example_intro_tga_data_analysis
   example_proximate_analysis
   example_oxidation_analysis
   example_soliddistillation
   example_kas_kinetics
   example_deconvolution
   example_reports
   tga
   kas_kinetics
   measure
   myfigure


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
