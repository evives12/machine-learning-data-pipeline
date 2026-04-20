import os
import matplotlib.pyplot as plt


def perform_eda(df):
    print("EDA Summary")
    print("-----------")
    print(f"Dataset Shape: {df.shape}\n")

    print("Columns:")
    print(df.columns.tolist())
    print()

    print("Summary Statistics:")
    print(df.describe())
    print()

    print("Survival Distribution:")
    print(df["Survived"].value_counts())

    # Create bar chart
    create_survival_chart(df)


def create_survival_chart(df):
    # Get project root
    base_dir = os.path.dirname(os.path.dirname(__file__))
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Count survival values
    counts = df["Survived"].value_counts()

    # Create plot
    plt.figure()
    counts.plot(kind="bar")

    plt.title("Survival Count")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Count")

    # Save chart
    file_path = os.path.join(reports_dir, "survival_distribution.png")
    plt.savefig(file_path)

    print(f"\nChart saved to: {file_path}")