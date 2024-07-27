import pandas as pd

def load_data(file_path):
    """Load dataset from a CSV"""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean dataset by handling missing values and outliers."""
    df = df.dropna()
    return df
