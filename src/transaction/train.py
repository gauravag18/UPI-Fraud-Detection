import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, IsolationForest
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

        smote = SMOTE(sampling_strategy=0.1, random_state=self.random_state)
        enn = EditedNearestNeighbours()

        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        X_clean, y_clean = enn.fit_resample(X_resampled, y_resampled)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_clean)
        X_test_scaled = scaler.transform(X_test)

        self.scaler = scaler
        return X_train_scaled, X_test_scaled, y_clean, y_test

    def evaluate(self, name, model, X_test, y_test):
        if name == "isolation_forest":
            y_prob = 1 - model.score_samples(X_test)
        else:
            y_prob = model.predict_proba(X_test)[:, 1]

        y_pred = (y_prob > 0.1).astype(int)

        pr_auc = average_precision_score(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        self.results[name] = {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "confusion_matrix": cm
        }

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
            max_depth=10,
            min_samples_leaf=50,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['rf'] = model
        self.evaluate("rf", model, X_test, y_test)

    def train_xgboost(self, X_train, X_test, y_train, y_test):
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            eval_metric='aucpr',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        self.evaluate("xgboost", model, X_test, y_test)

    def train_isolation_forest(self, X_train, X_test, y_train, y_test):
        model = IsolationForest(contamination=0.002)
        model.fit(X_train)
        self.models['isolation_forest'] = model
        self.evaluate("isolation_forest", model, X_test, y_test)

    def fit_all(self, path):
        print("Loading and preprocessing data\n")
        X, y = preprocess_pipeline(path)

        print(f"Dataset: {X.shape} | Fraud rate: {y.mean():.4f}")

        X_train, X_test, y_train, y_test = self.prepare_data(X, y)

        print("Training models : ")
        self.train_logistic(X_train, X_test, y_train, y_test)
        self.train_random_forest(X_train, X_test, y_train, y_test)
        self.train_xgboost(X_train, X_test, y_train, y_test)
        self.train_isolation_forest(X_train, X_test, y_train, y_test)

        return self.results