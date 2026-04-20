import pandas  as pd


def engineer_features(df):
    # Convert set to numeric(male =0, female =1)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # One-Hot endcaoke Embarked
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True).astype(int)

    return df