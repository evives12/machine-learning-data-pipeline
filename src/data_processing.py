def preprocess_data(df):
    # Drop columns that are not useful for basic modeling
    df = df.drop(columns=['Name', 'Ticket','Cabin' ])

    # Fill missing Afe values with Median
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Fill missing Embarked values with the mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Drop rows with missing Fare values, if any
    df = df.dropna(subset=['Fare'])

    return df