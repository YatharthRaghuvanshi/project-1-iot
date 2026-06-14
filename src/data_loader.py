import pandas as pd


def load_data(filepath):
    """
    Load AI4I Predictive Maintenance dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV dataset file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(filepath)


def clean_data(df):
    """
    Clean the dataset by removing duplicate rows and missing values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset.
    """

    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna()

    return df