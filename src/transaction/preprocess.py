import pandas as pd
import numpy as np
from src.transaction.feature_engineering import add_features

def load_data(path):
    df = pd.read_csv(path)
    return df

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
    df = df.dropna()
    return df

def process_time(df):
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.day
        df = df.drop(columns=["timestamp"])

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
        "transaction_status"
    ]

    existing_cols = [col for col in categorical_cols if col in df.columns]

    df = pd.get_dummies(df, columns=existing_cols, drop_first=True)

    return df

def split_features_target(df):
    df = df.copy()

    y = df["fraud_flag"]
    X = df.drop(columns=["fraud_flag"])

    return X, y

def preprocess_pipeline(path): 
    df = load_data(path) 
    df = clean_column_names(df) 
    df = basic_cleaning(df) 
    df = process_time(df) 
    df = add_features(df) 
    df = encode_categorical(df) 
    X, y = split_features_target(df) 
    return X, y