
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