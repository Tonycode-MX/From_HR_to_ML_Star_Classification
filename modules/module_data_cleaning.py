import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def nans_elimination(df):
    """
    Elimina filas con NaNs en el DataFrame.
    """
    if "target" not in df.columns:
        X = df
        y = None

        # Nans elimination
        imputer = SimpleImputer(strategy="mean")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        print("\nNaNs antes:", X.isna().sum().sum())
        print("NaNs después:", X_imputed.isna().sum().sum())
        df_cleaned = X_imputed
    else:
        # Features (X) y target (y)
        X = df.drop(columns=["target"])
        y = df["target"]

        # Nans elimination
        imputer = SimpleImputer(strategy="mean")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        print("\nNaNs antes:", X.isna().sum().sum())
        print("NaNs después:", X_imputed.isna().sum().sum())
        df_cleaned = pd.concat([X_imputed, y.reset_index(drop=True)], axis=1)

    return df_cleaned