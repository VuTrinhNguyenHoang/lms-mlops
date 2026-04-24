import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
