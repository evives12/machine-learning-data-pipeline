from data_loader import load_data
from src.data_processing import preprocess_data
from feature_engineering import engineer_features


def main():
  df = load_data()
  df = preprocess_data(df)

  print("Before Feature Engineering")
  print(df.head())

  df = engineer_features(df)

  print("\nAfter Feature Engineering")
  print(df.head())


if __name__ == '__main__':
    main()