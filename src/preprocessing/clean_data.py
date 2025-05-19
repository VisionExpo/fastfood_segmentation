import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

    # Label Encode Gender and VisitFrequency
    label_cols = ['Gender', 'VisitFrequency']
    for col in label_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    # Drop rows with missing or invalid data
    df_clean = df_clean.dropna()
    print(f"[INFO] Cleaned data shape: {df_clean.shape}")

    # Select numeric columns
    numeric_cols = yes_no_columns + ['Like', 'Age', 'VisitFrequency', 'Gender']
    X = df_clean[numeric_cols]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[INFO] Preprocessing complete. Data ready for clustering.")
    return X_scaled
