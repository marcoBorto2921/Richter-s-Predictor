# src/main.py
from data_utils import read_csv, display_data,encode_categorical,full_eda_report
from sklearn.model_selection import train_test_split
from model_utils import train_xgboost
import pandas as pd

def main():
    # Read data feauture
    X = read_csv("data/raw/train_values.csv")
    X_test = read_csv("data/raw/test_values.csv")

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
    X_test = encode_categorical(X_test)

    building_ids = X_test["building_id"].copy()
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y   
    )

    # Train model
    model = train_xgboost(X_train, y_train, X_test, y_test)

    # 4. Fai predizioni
    y_pred = model.predict(X_test)
    
    # Se avevi scalato o modificato le label (es. y-1), ricordati di tornare indietro
    y_pred = y_pred + 1
    
    # 5. Crea DataFrame con building_id e predizione
    submission = pd.DataFrame({
        "building_id": building_ids,
        "damage_grade": y_pred
    })
    
    # 6. Salva su CSV
    submission.to_csv("data/submission.csv", index=False)
    print("File submission.csv creato!")


if __name__ == "__main__":
    main()