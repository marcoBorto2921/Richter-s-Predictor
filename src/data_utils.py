# src/data_utils.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def read_csv(file_path):
    """Reads a CSV file and returns a DataFrame"""
    return pd.read_csv(file_path)

def display_data(df, n=10):
    """Prints the first n rows of the DataFrame in a readable way"""
    print(f"Here are the first {n} rows of the data:")
    print(df.head(n))  # stampa solo le prime n righe
    
    # Optionally print each row as a dictionary (first n rows)
    for idx, row in df.head(n).iterrows():
        print(f"Row {idx}: {row.to_dict()}")

def encode_categorical(df, method='onehot'):
    """
    Encode all categorical columns in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): the original DataFrame
        method (str): 'onehot' or 'label'
        
    Returns:
        pd.DataFrame: DataFrame with categorical columns encoded
    """
    categorical_cols = df.select_dtypes(include='object').columns
    if method == 'onehot':
        df = pd.get_dummies(df, columns=categorical_cols)
    elif method == 'label':
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    return df