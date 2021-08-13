# Quick and dirty setup of your python

Before installing `dive` and `deerlab`, see instructions in `dive` repository you can use the following to make sure you have your python environment set up correctly - important this will only work if youre using anaconda and if you have it set up.

1. Open a directory where the environment.yml file is located. The file is in this directory
2. Run: 
    ```bash
    conda env create -n pymc3 environment.yml
    ```
    This will create an environment with the name `pymc3` all required packages.
3. Use VS Code or any other editor to select this environment, or type 
    ```
    conda activate pymc3
    ```
    in your terminal. Then install `dive` and `deerlab` as instructed on the repos