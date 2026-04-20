from data_loader import load_data
from src.data_processing import preprocess_data
from feature_engineering import engineer_features
from model_training import train_model


def main():
  df = load_data()
  df = preprocess_data(df)
  df = engineer_features(df)

  print("Training Model...\n")

  model, X_test, y_test = train_model(df)

  print("Model trained successfully!")

  # Show sample predictions
  predictions = model.predict(X_test[:5])

  print("\nSample Predictions:")
  print(predictions)


if __name__ == '__main__':
    main()