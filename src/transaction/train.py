import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve,
    fbeta_score
)

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import joblib

from src.transaction.preprocess import preprocess_pipeline


class FraudDetector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}

        os.makedirs("models", exist_ok=True)

    def prepare_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

        #  LEAKAGE-FREE ENCODING
        # 1. Target Encoding (Smoothed) for high-cardinality columns
        target_cols = ["merchant_category", "sender_bank", "receiver_bank", "sender_state", "category_combo"]
        global_mean = y_train.mean()
        
        for col in target_cols:
            # Calculate means only from TRAIN
            agg = y_train.groupby(X_train[col]).agg(["count", "mean"])
            m = 100 
            smoothed = (agg["count"] * agg["mean"] + m * global_mean) / (agg["count"] + m)
            
            # Map to both sets
            X_train[f"{col}_risk"] = X_train[col].map(smoothed).fillna(global_mean)
            X_test[f"{col}_risk"] = X_test[col].map(smoothed).fillna(global_mean)
        
        # 2. One-Hot Encoding for remaining categorical columns
        cat_cols = [
            "transaction_type", "sender_age_group", "receiver_age_group", 
            "device_type", "network_type", "day_of_week", "transaction_status"
        ]
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
        
        # Align columns (in case some categories are missing in one set)
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

        # 3. Drop remaining raw string columns
        X_train = X_train.select_dtypes(exclude=['object'])
        X_test = X_test.select_dtypes(exclude=['object'])

        # 4. Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scaler = scaler
        return X_train_scaled, X_test_scaled, y_train, y_test

    def evaluate(self, name, model, X_test, y_test):
        y_prob = model.predict_proba(X_test)[:, 1]

        # OPTIMIZED THRESHOLDING: Maximize F-Beta (Beta=1.5) favoring Recall
        # While maintaining a reasonable precision (minimizing 30k clusters)
        
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        
        # Beta=1.5 gives more weight to recall than precision
        beta = 1.5
        f_beta = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls + 1e-10)
        
        # Find best threshold that keeps alert volume manageable (< 10% of test size)
        max_fp_threshold = np.percentile(y_prob, 90) # Top 10% volume
        
        valid_indices = np.where(thresholds >= max_fp_threshold)[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[np.argmax(f_beta[valid_indices])]
            best_threshold = thresholds[best_idx]
        else:
            best_threshold = thresholds[np.argmax(f_beta[:-1])]

        y_pred = (y_prob >= best_threshold).astype(int)

        pr_auc = average_precision_score(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        self.results[name] = {
            "accuracy": accuracy,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "confusion_matrix": cm
        }

        print(f"{name.upper()} Metrics :")
        print(f"Optimal Threshold: {best_threshold:.4f}")
        print(f"Accuracy: {accuracy:.4f} | PR AUC: {pr_auc:.4f} | ROC AUC: {roc_auc:.4f}")
        print(f"Precision: {report['1']['precision']:.4f} | Recall: {report['1']['recall']:.4f} | F1: {report['1']['f1-score']:.4f}")
        print("\n")

        # Save model
        joblib.dump(model, f"models/{name}.pkl")

    def train_logistic(self, X_train, X_test, y_train, y_test):
        model = LogisticRegression(class_weight='balanced', max_iter=2000)
        model.fit(X_train, y_train)
        self.models['logistic'] = model
        self.evaluate("logistic", model, X_test, y_test)

    def train_random_forest(self, X_train, X_test, y_train, y_test):
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=5,
            class_weight='balanced_subsample',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['rf'] = model
        self.evaluate("rf", model, X_test, y_test)

    def train_xgboost(self, X_train, X_test, y_train, y_test):
        # High scale_pos_weight to force attention on fraud
        scale_pos = 100 
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='aucpr',
            scale_pos_weight=scale_pos,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        self.evaluate("xgboost", model, X_test, y_test)

    def fit_all(self, path):
        print("Loading and preprocessing data\n")
        X, y = preprocess_pipeline(path)

        print(f"Dataset: {X.shape} | Fraud rate: {y.mean():.4f}")

        X_train, X_test, y_train, y_test = self.prepare_data(X, y)

        print("Training models : ")
        self.train_logistic(X_train, X_test, y_train, y_test)
        self.train_random_forest(X_train, X_test, y_train, y_test)
        self.train_xgboost(X_train, X_test, y_train, y_test)

        return self.results