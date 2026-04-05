import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from src.transaction.preprocess import preprocess_pipeline
from src.transaction.train import FraudDetector


def run_shap(path, model_name="xgboost", sample_size=50):
    """
    Explains model predictions using SHAP (SHapley Additive exPlanations).
    Reconstructs feature names to ensure plots are readable and identifies 
    which features contribute most to fraud detection.
    """
    # Load data
    print(f"\nSHAP Explainability for {model_name.upper()}:")
    X, y = preprocess_pipeline(path)

    # Use same preprocessing as training
    detector = FraudDetector()
    X_train, X_test, y_train, y_test = detector.prepare_data(X, y)

    # Convert y_test to Series if it's numpy from prepare_data (though it's returned as y_test from split)
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)

    # Load trained model
    model_path = f"models/{model_name}.pkl"
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found at {model_path}. Please train it first.")
        return

    model = joblib.load(model_path)

    # Take a sample for explanation (SHAP can be slow on large datasets)
    # Ensure we include fraud cases in the sample for more meaningful analysis
    fraud_indices = np.where(y_test == 1)[0]
    non_fraud_indices = np.where(y_test == 0)[0]
    
    # Stratified sample: prioritize fraud cases if available
    n_fraud = min(len(fraud_indices), sample_size // 2)
    n_normal = min(len(non_fraud_indices), sample_size - n_fraud)
    
    if len(fraud_indices) > 0 and len(non_fraud_indices) > 0:
        selected_indices = np.concatenate([
            np.random.choice(fraud_indices, n_fraud, replace=False),
            np.random.choice(non_fraud_indices, n_normal, replace=False)
        ])
    else:
        selected_indices = np.arange(min(sample_size, len(X_test)))
    
    X_sample_scaled = X_test[selected_indices]
    y_sample = y_test.iloc[selected_indices] if hasattr(y_test, 'iloc') else y_test[selected_indices]

    # RECONSTRUCT DataFrame with Feature Names for plot labels
    X_sample = pd.DataFrame(X_sample_scaled, columns=detector.feature_names)

    # Use TreeExplainer optimized for XGBoost
    print("Calculating SHAP values using TreeExplainer:")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # XGBoost binary classification often returns a single array for class 1
    # but sometimes (depending on version/params) a list [class_0, class_1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    # PLOTS 
    print("\n1. Generating SHAP Summary Plot")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_to_plot, X_sample, show=False)
    plt.title(f"SHAP Summary: {model_name.upper()} Fraud Drivers")
    plt.tight_layout()
    plt.show()

    print("\n2. Generating Feature Importance Bar Plot")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_to_plot, X_sample, plot_type="bar", show=False)
    plt.title(f"Global Feature Importance: {model_name.upper()}")
    plt.tight_layout()
    plt.show()

    # Print top contributing features summary
    global_importance = np.abs(shap_values_to_plot).mean(0)
    top_indices = np.argsort(global_importance)[::-1][:5]
    print("\nTOP 5 FRAUD INDICATORS (Global)")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {X_sample.columns[idx]:<30} | Magnitude: {global_importance[idx]:.4f}")

    # Single prediction explanation for the first Fraud case in sample
    fraud_cases_in_sample = np.where(y_sample == 1)[0]
    if len(fraud_cases_in_sample) > 0:
        fraud_idx = fraud_cases_in_sample[0]
        print(f"3. Explaining a specific Fraudulent transaction (Sample Index {fraud_idx}):")
        
        # Determine base value (expected value)
        if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) == 2:
            base_val = explainer.expected_value[1]
        else:
            base_val = explainer.expected_value

        plt.figure()
        shap.force_plot(
            base_val,
            shap_values_to_plot[fraud_idx],
            X_sample.iloc[fraud_idx],
            matplotlib=True,
            show=True
        )
    else:
        print("Note: No fraud cases were found in the explanation sample to show a single-case force plot.")