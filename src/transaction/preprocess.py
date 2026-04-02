import pandas as pd
from src.transaction.feature_engineering import create_advanced_features


def load_data(path):
    return pd.read_csv(path)


def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    return df


def basic_cleaning(df):
    df = df.copy()
    df = df.drop(columns=["transaction_id"], errors="ignore")
    return df


def encode_categorical(df):
    df = df.copy()

    categorical_cols = [
        "transaction_type",
        "merchant_category",
        "sender_age_group",
        "receiver_age_group",
        "sender_state",
        "sender_bank",
        "receiver_bank",
        "device_type",
        "network_type",
        "day_of_week",
        "transaction_status",
        "category_combo"
    ]

    existing_cols = [col for col in categorical_cols if col in df.columns]

    df = pd.get_dummies(df, columns=existing_cols, drop_first=True)

    return df


def split_features_target(df):
    df = df.copy()
    
    # remove timestamp (model can't use datetime)
    df = df.drop(columns=["timestamp"], errors="ignore")

    X = df.drop(columns=["fraud_flag"])
    y = df["fraud_flag"]

    return X, y


def preprocess_pipeline(path):
    df = load_data(path)
    df = clean_column_names(df)
    df = basic_cleaning(df)
    df = create_advanced_features(df)
    df = encode_categorical(df)
    X, y = split_features_target(df)

    return X, y