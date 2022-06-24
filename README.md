# Code for analyzing MinDE patterns

Cees Dekker Lab, Bionanoscience department, TU Delft, 03/2022, developed by Jacob Kerssemakers and Sabrina Meindlhumer

Preprint available at: https://www.biorxiv.org/content/10.1101/2022.04.28.489904v1

In the following, we provide instructions on how to set up a Python environment 'min_analysis' and install a package 'min_analysis_tools'. These tools can be used to perform analysis on Min protein surface pattern data.
The folder 'min_analysis_scripts' contains scripts that use the provided tools to perform analysis on single stacks or perform batch processing of all stacks within a given folder.

This repository contains three main notebooks:
- DEMO_MinDE_global_analysis.ipynb: demonstration of global analysis
- DEMO_MinDE_local_analysis.ipynb: demonstration of local analysis
- Quickstart_Min_analysis.ipynb: quick analysis tool for single-stack Min pattern analysis
Further, a series of "Supplementary" notebooks, which provide accuracy tests and controls for the provided methods.

Jupyter notebooks can be accessed in the following binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/M-Sabrina/MinDE_analysis_2022/HEAD)

The first release can be found at Zenodo:

[![DOI](https://zenodo.org/badge/474316785.svg)](https://zenodo.org/badge/latestdoi/474316785)


# Installation

- Install miniconda from https://docs.conda.io/en/latest/miniconda.html 
- Open terminal (on Windows, search for 'Anaconda prompt')
- Install Git from https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
- Clone this repository with `git clone`
- Within a terminal, change directory into this repository (`cd mypath`, replace mypath with the directory in which environment.yml is located)
- Create the environment by executing `conda env create -f environment.yml`
- Install the Python package by executing `pip install -e .`

# Execute Python files

- Activate the environment by executing `conda activate min_analysis` in a terminal
- Can now run python file by executing `python NAME_OF_FILE.py`

Alternatively, change the interpreter path of your IDE to the `min_analysis` environment.
For Spyder, that can be done as follows:
- Go to Preferences
- Go to tab 'Python interpreter'
- Select 'Use the following Python interpreter'
- With Miniconda and on Windows, the path to the Python interpreter will be as follows: `C:/Users/USERNAME/Miniconda3/envs/min_analysis/python.exe` (change `USERNAME` to your user)

# Execute Jupyter notebook

- Activate the environment by executing `conda activate min_analysis` in a terminal
- Execute `jupyter lab`
- Open the notebook (`.ipynb` extension) within Jupyter

# Remove environment

If you want to remove the environment (because you don't need it anymore or you want to recreate it), execute `conda env remove -n min_analysis` from the base environment.
