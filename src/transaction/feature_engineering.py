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

    #  TIME FEATURES 
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["night_risk"] = ((df["hour"] >= 22) | (df["hour"] <= 4)).astype(int)

    #  BANK FEATURES
    df["cross_bank"] = (df["sender_bank"] != df["receiver_bank"]).astype(int)

    # Bank pair rarity
    pair_counts = df.groupby(["sender_bank", "receiver_bank"]).size()
    sender_counts = df["sender_bank"].value_counts()

    def get_pair_rarity(row):
        pair = (row["sender_bank"], row["receiver_bank"])
        return pair_counts.get(pair, 0) / sender_counts.get(row["sender_bank"], 1)

    df["bank_pair_rarity"] = df.apply(get_pair_rarity, axis=1)

    #  VELOCITY 
    df = df.sort_values("timestamp")
    df["sender_velocity"] = df.groupby("sender_bank").cumcount()

    #  SAFE CATEGORY ENCODING (NO LEAKAGE) 
    df["category_combo"] = df["merchant_category"].astype(str) + "_" + df["transaction_type"].astype(str)

    #  INTERACTIONS 
    df["amount_night"] = df["amount_log"] * df["night_risk"]
    df["bank_amount"] = df["cross_bank"] * df["high_amount"]

    #  FINAL SCORE (SAFE, NO FRAUD LABEL USED) 
    df["master_risk_score"] = (
        df["high_amount"] * 0.3 +
        df["night_risk"] * 0.2 +
        df["cross_bank"] * 0.2 +
        df["bank_pair_rarity"] * 0.15 +
        df["micro_amount"] * 0.1 +
        df["sender_velocity"] * 0.05
    )

    return df