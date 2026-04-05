from src.transaction.train import FraudDetector
import pandas as pd
from src.transaction.shap_explain import run_shap


def print_results(results):
    df = pd.DataFrame(results).T

    print("Model Comparison: \n")

    print(df[["pr_auc", "roc_auc", "precision", "recall", "f1"]].round(4))

    print("\nConfusion Matrices:")
    for model, res in results.items():
        print(f"\n{model.upper()}:")
        print(res["confusion_matrix"])


def main():
    DATA_PATH="transaction_data/upi_transactions_2024.csv"

    # Train models
    detector = FraudDetector()
    results = detector.fit_all(DATA_PATH)

    # Print results
    print_results(results)

    # SHAP Explainability
    run_shap(DATA_PATH, model_name="xgboost")

if __name__ == "__main__":
    main()