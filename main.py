from src.transaction.train import FraudDetector
import pandas as pd

def print_results(results):
    df = pd.DataFrame(results).T

    print("FINAL MODEL COMPARISON")

    print(df[["pr_auc", "roc_auc", "precision", "recall", "f1"]].round(4))

    print("\nConfusion Matrices:")
    for model, res in results.items():
        print(f"\n{model.upper()}:")
        print(res["confusion_matrix"])


def main():
    DATA_PATH = "transaction_data/upi_transactions_2024.csv"

    detector = FraudDetector()
    results = detector.fit_all(DATA_PATH)

    print_results(results)


if __name__ == "__main__":
    main()