import pandas as pd

def load(filename: str) -> pd.Series:
    df = pd.read_csv(filename, header=None)
    close_column = df.iloc[:, 1]
    close_column = close_column.str.replace(',', '.')
    close_column = pd.to_numeric(close_column, errors='coerce')
    close_column = close_column.dropna()
    return close_column
