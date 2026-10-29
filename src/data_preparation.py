import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, sep=";", decimal=",")

    # Remove completely empty rows and columns
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    # Remove invalid measurement markers with missing values
    df = df.replace(-200, np.nan)

    return df


def chronological_train_test_split(df, target="NO2(GT)", train_size=0.8):
    split_index = int(len(df) * train_size)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # Exclude rows without a valid regression target
    train_df = train_df.dropna(subset=[target]).copy()
    test_df = test_df.dropna(subset=[target]).copy()

    return train_df, test_df

