import pandas as pd
import numpy as np

def create_advanced_features(df):
    df = df.copy()

    # TIME 
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    #  AMOUNT FEATURES 
    df["amount_log"] = np.log1p(df["amount_inr"])
    df["amount_sqrt"] = np.sqrt(df["amount_inr"])

    df["high_amount"] = (df["amount_inr"] > df["amount_inr"].quantile(0.98)).astype(int)
    df["micro_amount"] = (df["amount_inr"] < df["amount_inr"].quantile(0.02)).astype(int)

    #  ADVANCED RATIOS & FREQUENCIES
    # Merchant stats
    df["merchant_amt_mean"] = df.groupby("merchant_category")["amount_inr"].transform("mean")
    df["merchant_amt_std"] = df.groupby("merchant_category")["amount_inr"].transform("std").fillna(1)
    df["merchant_zscore"] = (df["amount_inr"] - df["merchant_amt_mean"]) / (df["merchant_amt_std"] + 1e-5)
    
    # Bank stats
    df["bank_amt_mean"] = df.groupby("sender_bank")["amount_inr"].transform("mean")
    df["bank_amt_std"] = df.groupby("sender_bank")["amount_inr"].transform("std").fillna(1)
    df["bank_zscore"] = (df["amount_inr"] - df["bank_amt_mean"]) / (df["bank_amt_std"] + 1e-5)
    
    # Frequency features
    df["merchant_freq"] = df.groupby("merchant_category")["amount_inr"].transform("count") / len(df)
    df["sender_bank_freq"] = df.groupby("sender_bank")["amount_inr"].transform("count") / len(df)
    df["receiver_bank_freq"] = df.groupby("receiver_bank")["amount_inr"].transform("count") / len(df)

    # Category combo for string interaction
    df["category_combo"] = df["merchant_category"].astype(str) + "_" + df["transaction_type"].astype(str)

    #  TIME FEATURES (Cyclical)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    #  BANK INTERACTIONS
    df["is_cross_bank"] = (df["sender_bank"] != df["receiver_bank"]).astype(int)

    #  VELOCITY
    # Number of transactions in the same hour for this merchant/bank pair
    df["merchant_hour_volume"] = df.groupby(["merchant_category", "hour"])["amount_inr"].transform("count")
    
    #  INTERACTIONS
    df["high_amount_night"] = ((df["amount_inr"] > 2000) & (df["is_night"] == 1)).astype(int)

    #  FINAL CLEANUP
    # Drop string-based segment to avoid dummy-variable explosion
    cols_to_drop = ["merchant_amt_mean", "merchant_amt_std", "bank_amt_mean", "bank_amt_std", "user_segment"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df