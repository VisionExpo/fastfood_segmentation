import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def preprocess_data(df):
    """
    Preprocess the fast food dataset:
    - Convert Yes/No to 1/0
    - Label encode Gender and VisitFrequency
    - Scale all numeric features

    Args:
        df (pd.DataFrame): Raw input data

    Returns:
        np.ndarray: Scaled numeric features for clustering
    """
    print("[INFO] Starting preprocessing...")

    df_clean = df.copy()

    # Convert Yes/No to 1/0
    yes_no_columns = [
        'yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast',
        'cheap', 'tasty', 'expensive', 'healthy', 'disgusting'
    ]
    df_clean[yes_no_columns] = df_clean[yes_no_columns].replace({'Yes': 1, 'No': 0})

    # Clean the 'Like' column
    if 'Like' in df_clean.columns:
        print("[INFO] Cleaning 'Like' column...")
        like_mapping = {
            'I HATE IT!-5': -5, '-5': -5,
            '-4': -4,
            '-3': -3,
            '-2': -2,
            '-1': -1,
            '0': 0,
            '+1': 1, '1': 1,
            '+2': 2, '2': 2,
            '+3': 3, '3': 3,
            '+4': 4, '4': 4,
            'I LOVE IT!+5': 5, '+5': 5,
            'I love it!+5': 5  # Handling the specific problematic value
        }
        df_clean['Like'] = df_clean['Like'].replace(like_mapping)
        df_clean['Like'] = pd.to_numeric(df_clean['Like'], errors='coerce') # Convert to numeric, errors become NaN
    else:
        print("[WARNING] 'Like' column not found.")

    # Label Encode Gender
    if 'Gender' in df_clean.columns:
        le = LabelEncoder()
        df_clean['Gender'] = le.fit_transform(df_clean['Gender'].astype(str))
    else:
        print("[WARNING] 'Gender' column not found.")

    # Ordinal Encode 'VisitFrequency'
    if 'VisitFrequency' in df_clean.columns:
        print("[INFO] Ordinally encoding 'VisitFrequency' column...")
        visit_freq_mapping = {
            'Never': 0,
            'Once a year': 1,
            'Every three months': 2,
            'Once a month': 3,
            'Once a week': 4,
            'More than once a week': 5
        }
        df_clean['VisitFrequency'] = df_clean['VisitFrequency'].replace(visit_freq_mapping)
        df_clean['VisitFrequency'] = pd.to_numeric(df_clean['VisitFrequency'], errors='coerce') # Convert to numeric, errors become NaN
    else:
        print("[WARNING] 'VisitFrequency' column not found.")

    # Select numeric columns for scaling (ensure 'Age' is present and numeric)
    # 'Like', 'VisitFrequency' are now numeric or NaN
    numeric_cols = yes_no_columns + ['Like', 'Age', 'VisitFrequency', 'Gender']
    # Filter out any columns that might not exist or are not fully numeric
    X = df_clean[[col for col in numeric_cols if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col])]]

    # Drop rows with any NaN values in the selected columns (X)
    X = X.dropna()
    print(f"[INFO] Data shape after NaN removal and column selection for scaling: {X.shape}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[INFO] Preprocessing complete. Data ready for clustering.")
    return X_scaled
