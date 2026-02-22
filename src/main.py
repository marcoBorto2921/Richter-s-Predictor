# src/main.py
from data_utils import read_csv, display_data,encode_categorical
from sklearn.model_selection import train_test_split


def main():
    # Read data feauture
    X = read_csv("data/raw/train_values.csv")

    # Read labels
    y = pd.read_csv("data/raw/train_labels.csv")
    
    # Display data feature
    display_data(X)

    # Data feature to numerical data 
    X = encode_categorical(df)

    # Split training and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


if __name__ == "__main__":
    main()