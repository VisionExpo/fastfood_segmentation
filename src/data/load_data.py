
import pandas as pd

def load_fastfood_data(filepath):
    """
    Loads the fast food dataset from the given CSV path.

    Args:
        filepath (str): Relative or absolute path to the dataset CSV.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        raise
    except Exception as e:
        print(f"[ERROR] An error occurred while loading the data: {e}")
        raise
