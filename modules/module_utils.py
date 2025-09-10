import numpy as np
import pandas as pd
import os

def save_dataframe(df, data_folder, filename="dataframe.csv"):
    """
    Saves the DataFrame to a CSV file at the specified path.
    """
    save_path = os.path.join(data_folder, filename)
    df.to_csv(save_path, index=False)
    print(f"\nDataFrame saved to {save_path}")

def add_color_magnitude_indices(df):
    """
    Adds color and magnitude indices to the DataFrame.

    Color index: BP_RP = phot_bp_mean_mag - phot_rp_mean_mag
    Magnitude index: M_G = phot_g_mean_mag + 5 - 5*log10(d_pc), where d_pc = 1000/parallax
    """
    df["BP_RP"] = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]
    d_pc = 1000.0 / df["parallax"]   # distancia en parsecs (parallax en mas)
    df["M_G"] = df["phot_g_mean_mag"] + 5 - 5*np.log10(d_pc)
    return df

def label_star(row):
    """
    Labels a star based on its color and magnitude indices.
    Criteria:
      - White Dwarf (WD): -0.5 <= BP_RP <= 1.8 and M_G >= 10
      - Giant: BP_RP >= 0.8 and M_G <= 2.5
      - Main Sequence (MS): all other cases
    """
    # White Dwarfs: blue and faint
    if (-0.5 <= row.BP_RP <= 1.8) and (row.M_G >= 10):
        return "WD"      # White Dwarf
    # Giants: red and bright
    elif (row.BP_RP >= 0.8) and (row.M_G <= 2.5):
        return "Giant"
    # Main Sequence: intermediate
    else:
        return "MS"      # Main Sequence
    
def star_counts(df):
    """
    Prints the counts of each star type in the DataFrame.
    """
    print("\n=================")
    print("Star type counts:")
    print("=================")
    print(df["target"].value_counts())
    print("=================\n")