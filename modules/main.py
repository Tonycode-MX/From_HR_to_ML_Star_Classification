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
from modules.module_data_cleaning import nans_elimination
from data.data_import import gaia_data_import
from modules.module_utils import add_color_magnitude_indices, label_star, star_counts, save_dataframe

# Number of stages to execute (1 to )
stages = [1]

# STAGE 1: Data Import and cleaning
def stage1():
    """
    Main script to choose how to import the dataset.
    Uncomment the preferred option:
      1) Import directly from Gaia archive (astroquery)
      2) Import from local CSV file
    """

    # Get data path
    data_folder = data_path()

    # OPTION 1: Import from Gaia
    df = gaia_data_import()  # guest login by default

    # OPTION 2: Import from local CSV in data folder
    #data_path = data_path()
    #dataset = os.path.join(data_path, "dataset.csv")
    #df = pd.read_csv(dataset)

    # Data cleaning
    df_cleaned = nans_elimination(df)

    # Save the imported DataFrame as CSV in /data
    save_dataframe(df_cleaned, data_folder, filename="cleaned_data.csv")

    # Preview
    print("\nImported dataset preview:")
    print(df_cleaned.head())

#STAGE 2: Data Classification
def stage2():
    """
    Main script for classification of the dataset.
    """
    # Get data path
    data_folder = data_path()
    dataset = os.path.join(data_folder, "cleaned_data.csv")
    df = pd.read_csv(dataset)

    # Add color and magnitude indices
    df = add_color_magnitude_indices(df)

    # Label stars
    df["target"] = df.apply(label_star, axis=1)

    # Star counts
    star_counts(df)

    # Save the imported DataFrame as CSV in /data
    save_dataframe(df, data_folder, filename="classified_dataset.csv")

    # Preview
    print("\nClassified dataset preview:")
    print(df.head())


# Execute stages
if __name__ == '__main__': 
    
    if 1 in stages:
        stage1()
    elif 2 in stages:
        stage2()
