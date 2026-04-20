from data_loader import load_data
from data_processing import preprocess_data
from feature_engineering import engineer_features
from model_training import train_model
from evaluation import evaluate_model
from eda import perform_eda


def main():
    df = load_data()

    print("Running EDA on raw data...\n")
    perform_eda(df)

    df = preprocess_data(df)
    df = engineer_features(df)

    print("\nTraining model...\n")
    model, X_test, y_test = train_model(df)

    accuracy, precision, mse = evaluate_model(model, X_test, y_test)

    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"MSE: {mse:.2f}")


if __name__ == "__main__":
    main()