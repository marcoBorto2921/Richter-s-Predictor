# src/data_utils.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import os

def read_csv(file_path):
    """Reads a CSV file and returns a DataFrame"""
    return pd.read_csv(file_path)

def display_data(data, n=10):
    """Display first n rows of DataFrame or Series"""
    print(f"\nHere are the first {n} rows:")

    print(data.head(n))

    if isinstance(data, pd.DataFrame):
        for idx, row in data.head(n).iterrows():
            print(f"Row {idx}: {row.to_dict()}")
    else:  # Series
        for idx, value in data.head(n).items():
            print(f"Index {idx}: {value}")


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

    # Save preprocessed features
    X.to_csv("data/preprocessed/train_values_preprocessed.csv", index=False)
    return df


def full_eda_report(df, y):
    """
    Full exploratory data analysis report.
    
    Parameters:
        df (pd.DataFrame): feature dataframe
        y (pd.Series): target labels
    """

    print("\n==================== DATASET INFO ====================")
    print(df.info())

    print("\n==================== NUMERIC STATISTICS ====================")
    print(df.describe())

    print("\n==================== MISSING VALUES ====================")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")

    # -------------------------
    # LABEL DISTRIBUTION
    # -------------------------
    print("\n==================== LABEL DISTRIBUTION ====================")
    print("Counts:")
    print(y.value_counts())

    print("\nPercentages:")
    print((y.value_counts(normalize=True) * 100).round(2))

    plt.figure(figsize=(6,4))
    sns.countplot(x=y)
    plt.title("Label Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig("data/eda/label_distribution.png")
    plt.close()

    # -------------------------
    # NUMERIC DISTRIBUTIONS
    # -------------------------
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()

    # -------------------------
    # CORRELATION MATRIX
    # -------------------------
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10,8))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()