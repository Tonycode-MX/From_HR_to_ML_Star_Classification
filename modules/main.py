# Main module for executing stages

# LIBRARIES

# Standard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# External modules
from modules.module_data_path import data_path, plot_data_path
from data.data_import import gaia_data_import

# Number of stages to execute (1 to )
stages = [1]

# STAGE 1: Data Import
def stage1():

    """
    Main script to choose how to import the dataset.
    Uncomment the preferred option:
      1) Import directly from Gaia archive (astroquery)
      2) Import from local CSV file
    """

    # OPTION 1: Import from Gaia
    df = gaia_data_import()  # guest login by default

    # # (Optional) Save the imported DataFrame as CSV in /data
    # save_path = os.path.join(data_path, "dataset_from_gaia.csv")
    # df.to_csv(save_path, index=False)
    # print(f"Dataset saved to {save_path}")

    # OPTION 2: Import from local CSV in data folder
    #data_path = data_path()
    #dataset = os.path.join(data_path, "dataset.csv")
    #df = pd.read_csv(dataset)

    # Preview
    print("\nImported dataset preview:")
    print(df.head())


# Execute stages
if __name__ == '__main__': 
    
    if 1 in stages:
        stage1()
