# src/data_utils.py
import pandas as pd

def read_csv(file_path):
    """Reads a CSV file and returns a DataFrame"""
    return pd.read_csv(file_path)

def display_data(df):
    """Prints the DataFrame in a readable way"""
    print("Here are the data:")
    print(df)
    
    # Optionally print each row as a dictionary
    for idx, row in df.iterrows():
        print(f"Row {idx}: {row.to_dict()}")