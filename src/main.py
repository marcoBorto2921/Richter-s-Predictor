# src/main.py
from data_utils import read_csv, display_data,encode_categorical,full_eda_report
from sklearn.model_selection import train_test_split
from model_utils import train_xgboost

def main():
    # Read data feauture
    X = read_csv("data/raw/train_values.csv")

    # Read labels
    y = read_csv("data/raw/train_labels.csv")["damage_grade"]
    y = y - 1
    
    # Display data feature
    # display_data(X)
    # display_data(y)

    # Statistical analysis
    # full_eda_report(X, y)

    # Data feature to numerical data 
    X = encode_categorical(X)
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y   
    )

    # Train model
    model = train_xgboost(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()