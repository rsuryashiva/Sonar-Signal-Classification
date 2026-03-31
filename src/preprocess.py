import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

def load_data(path):
    df = pd.read_csv(path, header=None)
    df[60] = df[60].map({"R": 0, "M": 1})
    return df

def split_data(df):
    X = df.drop(columns=[60])
    y = df[60]

    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler