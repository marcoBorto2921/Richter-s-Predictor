# src/main.py
from data_utils import read_csv, display_data

def main():
    # Read data
    df = read_csv("data/raw/test_values.csv")
    
    # Display data
    display_data(df)

if __name__ == "__main__":
    main()