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
from modules.module_models import prepare_data_for_modeling, rf_feature_selection, rfe_feature_selection, compare_feature_selection, compare_model_performance

# Number of stages to execute (1 to 3)
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

    # Preview (optional)
    #print("\nImported dataset preview:")
    #print(df_cleaned.head())

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

    # Preview (optional)
    #print("\nClassified dataset preview:")
    #print(df.head())

# STAGE 3: RF & REF-based Modeling
def stage3():
    """
    Main script for feature selection using Random Forest and RFE.
        1) Prepare data for modeling (feature selection, train-test split, encoding).
        2) Feature selection with Random Forest.
        3) Feature selection with RFE.
        4) Compare feature selection results.
        5) Comparison of training with all features vs. K features and validation.
    """
    # Get data path
    data_folder = data_path()
    dataset = os.path.join(data_folder, "classified_dataset.csv")
    df = pd.read_csv(dataset)

    n_ft = 5  # Number of top features to select

    # Prepare data for modeling
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_modeling(df)

    # Feature selection with Random Forest
    topK_rf = rf_feature_selection(X_train, y_train, n_features=n_ft, show_importance=False)  #show_importance=True to print importances

    # Feature selection with RFE
    topK_rfe = rfe_feature_selection(X_train, y_train, n_features=n_ft)

    # Compare feature selection results (optional)
    #compare_feature_selection(topK_rf, topK_rfe)

    # Comparison of training with all features vs. K features and validation.
    compare_model_performance(X_train, y_train, X_val, y_val, X_test, y_test, selected_features=topK_rfe)


# Execute stages
if __name__ == '__main__': 
    
    if 1 in stages:
        stage1()
    elif 2 in stages:
        stage2()
    elif 3 in stages:
        stage3()
