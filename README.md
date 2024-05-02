# Thermogravimetric Analysis in Python

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![Python 3.10](https://img.shields.io/badge/python-3.10%2B-blue)

[![Testing (CI)](https://github.com/mpecchi/tga_data_analysis/actions/workflows/continuous_integration.yaml/badge.svg)](https://github.com/mpecchi/tga_data_analysis/actions/workflows/continuous_integration.yaml)

[![Publishing (CI)](https://github.com/mpecchi/tga_data_analysis/actions/workflows/python-publish.yaml/badge.svg)](https://github.com/mpecchi/tga_data_analysis/actions/workflows/python-publish.yaml)


The `tga_data_analysis` tool automates the typical analysis of thermogravimetric analysis (TGA) data, saving time, avoiding human error, and increasing comparability of results from different groups. 

## Framework

### File

A **.txt** or **.csv** file located in the project folder that contains **time**, **temperature**, and **mass loss** information for a measure.

Depending on the instrument export parameter, the structure of ``Files`` can differ slightly. ``Project-Parameters`` ensure that the loading process can lead to the same data structure to allow to perform all downstream computations.

A good naming convention for ``Files`` consists in using ``_`` to ONLY indicate replicates ("A_1", "A_2", or long-sample-name_1, not "name_with_underscores_1").

### Sample

A collection of ``Files`` that replicate the same measure and ensure reproducibility.

If the ``Project-Parameters`` do not apply to a specific ``Sample``, their values can be modified for a single ``Sample`` instance.

The ``Files`` in the ``Sample`` are identified by their names and loaded.

Each numerical (ex. ash) or array value (ex. the time vector) from each ``Files`` is stored as a **replicate** using the ``Measure`` class, which provides access to each replicate of the value but also to **average** and **standard deviation** for each value. 

The mass loss profile for each replicate are projected on a common temperature vector thus avoiding asynchronies and artifact peaks in the average values due to instrumental micro-delays. The original temperature, time, and mass loss vector are stored for each ``File``.

``Single-sample Analyses`` methods are provided to perform common TGA data analysis at the ``Sample`` level:

* ``Proximate Analysis``: determines the moisture, volatile matter, and ash content from TGA data.

* ``Oxidation Analysis``: Analyzes the oxidation behavior of materials.

* ``Solid-Distillation Analysis``: Studies the thermal decomposition and distillation characteristics of solids.

* ``Peak Deconvolution Analysis``: Resolves overlapping thermal decomposition events.

The ``Sample`` class can generate ``multi-replicate reports`` and ``multi-replicate plots`` for TG and DTG curves and for the results of any of the ``Single-sample Analyses``.

### Project

The ``folder path`` indicates where the ``Files`` are located and where the ``output`` folder will be created.

The ``Project-Parameters`` are valid for each ``Sample`` unless specified at the ``Sample`` initialization.

``Samples`` can be added using the ``add_sample`` method or by specifying the ``Project`` to a new ``Sample`` instance during initialization.

The ``Project`` can generate reports and plots using the following methods:

* ``multireport``: Generate a multi-sample report based on the specified report type and style

* ``plot_multi_tg``: Plot multiple thermogravimetric (TG) curves for the given samples.

* ``plot_multi_dtg``: Plot multiple derivative thermogravimetric (DTG) curves for the given samples.

* ``plot_multi_soliddist``: Plot multiple solid distribution curves for the given samples.

* ``plot_multireport``: Plot the results for the multi-sample report

### Multi-sample Analyses

For analysis that require data from multiple samples (ex. KAS kinetics), a multi-sample class that includes multiple ``Sample`` objects is defined (ex. ``KasSample``).

Multi-sample classes provide the methods to perform the dedicated analysis and plot the results.
The available ones are 

* ``KAS Kinetic Analysis``: Applies the Kissinger-Akahira-Sunose method to determine kinetic parameters.

### Project-Sample-Parameters
If specified at the ``Project`` level become the default for all ``Samples`` and therefore ``Files``. They can be overwritten for each single ``Sample`` instance. The most important are described here, see the docs for the rest.

* ``load_skiprows`` an int that indicates the number of rows that must be skipped in the file at loading. The first valid row should be the one that contains the name of the columns ("time", "temperature", "tg"; these are just examples).

* ``column_name_mapping`` a dictionary used to specify how to rename the columns in the ``File`` to the standard names that the software can reliably use. These names are ``t_min``, ``T_C``, ``m_p``, and ``m_mg`` for time, temperature, mass percentage, and mass in mg, respectively. At least the first three must be present (if m_mg is missing, it is assumed to be equal to m_p).

* ``time_moist``: The time in minutes where the mass loss should be considered moisture.

* ``time_vm``: The time in minutes where the mass loss should be considered volatile matter.

* ``temp_initial_celsius``: The initial temperature where every ``File`` is going to start to ensure uniformity.

* ``temp_lim_dtg_celsius``: The temperature limits for DTG analysis, in Celsius. It should exclude moisture and fixed 
carbon segments.

* ``temp_unit``: The unit of temperature the project will convert everything to, not the unit in the ``Files``. 

* ``resolution_sec_deg_dtg``: The resolution in seconds per degree for the common temperature vector used in the DTG 
analysis.

* ``dtg_window_filter``: The window size for the Savitzky-Golay filter used to smooth the DTG curve.

* ``temp_i_temp_b_threshold``: The fractional threshold for the detection of Ti (t_ignition) and Tb (burnout) calculation in DTG analysis.

**Example**

If files start with 10 method rows before the real data and the columns are names "time/minutes", "temp/C", and "m/%",
then the ``Project-Parameters`` should be:
```bash
load_skiprows=10
column_name_mapping={"time/minutes": "t_min", "temp/C": "T_C", "m/%": "m_p"}
```


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


## Nomenclature

* ar: as received
* db: dry basis
* daf: dry, ash-free
* vm: volatile matter
* fc: fixed carbon

## Plotting with myfigure

Plots rely on the package ``myfigure``, a package to simplify 
Check out the [documentation](https://myfigure.readthedocs.io/) and 
[GitHub](https://github.com/mpecchi/myfigure/).
