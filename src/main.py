from data_loader import load_data
from src.data_processing import preprocess_data
from feature_engineering import engineer_features
from model_training import train_model
from evaluation import evaluate_model


def main():
  df = load_data()
  df = preprocess_data(df)
  df = engineer_features(df)

  print("Training Model...\n")

  model, X_test, y_test = train_model(df)

  print("Model trained successfully!\n")

  accuracy, precision, mse = evaluate_model(model, X_test, y_test)

  print("Model Evaluation: ")
  print(f"Accuracy: {accuracy:.2f}")
  print(f"Precision: {precision:.2f}")
  print(f"MSE: {mse:.2f}")


if __name__ == '__main__':
    main()