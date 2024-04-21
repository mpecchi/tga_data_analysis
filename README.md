# Thermogravimetric Analysis in Python

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![Python 3.10](https://img.shields.io/badge/python-3.10%2B-blue)

[![Testing (CI)](https://github.com/mpecchi/tga_data_analysis/actions/workflows/continuous_integration.yaml/badge.svg)](https://github.com/mpecchi/tga_data_analysis/actions/workflows/continuous_integration.yaml)

[![Publishing (CI)](https://github.com/mpecchi/tga_data_analysis/actions/workflows/python-publish.yaml/badge.svg)](https://github.com/mpecchi/tga_data_analysis/actions/workflows/python-publish.yaml)


The `tga_data_analysis` tool automates the typical analysis of thermogravimetric analysis (TGA) data, saving time, avoiding human error, and increasing comparability of results from different groups. 

## Framework

### Project

The ``Project`` class refers to a folder that containes all TGA data for a project and to a set of project-level parameters (example temperature unit and plot style). 

TGA files that are replicates of the same analysis are loaded and treated as a single ``Samples``. 

The ``Project`` class can generate publication quality **multi-sample reports** and **multi-sample plots**.


### Sample

A collection of replicate runs (TGA files) of the same test, specified by their filenames.

The ``Sample`` class provides direct access to single replicate results and to their average and standard deviation. 

The ``Measure`` class is used to store each numerical value, so that single replicate, average, and standard deviation are available for each intermediate step and final result.

The ``Sample`` class can generate **multi-replicate reports** and **multi-replicate plots** for data inspection.

## Single-sample Analyses

The ``Sample`` class provides method to perform common TGA data analysis at the sample level, providing statistics based on the replicates.

### Proximate Analysis
Determines the moisture, volatile matter, and ash content from TGA data.
[![Pic](https://github.com/user/repository/blob/main/path/to/small-image.png?raw=true)](https://github.com/user/repository/blob/main/path/to/full-size-image.png?raw=true)
### Oxidation Analysis
Analyzes the oxidation behavior of materials.


### Solid-Distillation Analysis
Studies the thermal decomposition and distillation characteristics of solids.


### Peak Deconvolution Analysis
Resolves overlapping thermal decomposition events.


## Multi-sample Analyses

For analysis that require data from multiple samples (ex. KAS kinetics), a multi-sample class that includes multiple ``Sample`` objects is defined (ex. ``KasSample``).

Multi-sample classes provide the methods to perform the dedicated analysis and plot the results.

### KAS Kinetic Analysis
Applies the Kissinger-Akahira-Sunose method to determine kinetic parameters.

## Documentation

Check out the [documentation](https://tga-data-analysis.readthedocs.io/).

## Installation

You can install the package from [PyPI](https://pypi.org/project/tga_data_analysis/):

```bash
pip install tga_data_analysis
```

## Examples

Each example is available as a folder in the ``examples`` folder and contains the code and the necessary input data.
To run examples:
1. Install ``tga_data_analysis`` in your Python environment
2. Download the folder that contains the example
3. Run the code 
4. If you run the scripts as Jupyter Notebooks, replace the relative path at the beginning of each example with the absolute path to the folder where the code is located 
