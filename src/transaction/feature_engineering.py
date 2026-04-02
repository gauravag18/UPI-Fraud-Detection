import pandas as pd
import numpy as np

def add_features(df):
    df = df.copy()

    if "amount_inr" in df.columns:
        df["is_high_amount"] = (df["amount_inr"] > df["amount_inr"].quantile(0.95)).astype(int)

    if "hour" in df.columns:
        df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)

    if "sender_bank" in df.columns and "receiver_bank" in df.columns:
        df["is_cross_bank"] = (df["sender_bank"] != df["receiver_bank"]).astype(int)

    if "network_type" in df.columns:
        df["is_low_network"] = df["network_type"].isin(["3G"]).astype(int)

    if "transaction_status" in df.columns:
        df["is_failed"] = (df["transaction_status"] == "FAILED").astype(int)

    return df