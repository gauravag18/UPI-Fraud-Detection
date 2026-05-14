import pandas as pd
from src.transaction.train       import FraudDetector
from src.transaction.shap_explain import run_shap


DATA_PATH  = "transaction_data/upi_transactions_2024.csv"
MODEL_DIR  = "models"
DOCS_DIR   = "docs"
BEST_MODEL = "xgboost"


def print_results(results: dict) -> None:
    df   = pd.DataFrame(results).T
    cols = [c for c in ["recall", "precision", "f1", "f2", "pr_auc", "roc_auc"]
            if c in df.columns]

    print("\n" + "=" * 65)
    print("  MODEL COMPARISON")
    print("=" * 65)
    print(df[cols].round(4).to_string())

    print("\n  CONFUSION MATRICES:")
    for model, res in results.items():
        cm = res["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel()
        print(f"\n  {model.upper()}")
        print(f"    TP (Fraud caught)   : {tp}")
        print(f"    FN (Fraud missed)   : {fn}")
        print(f"    FP (False alarms)   : {fp}")
        print(f"    TN (Legit correct)  : {tn}")
        catch_rate = tp / max(tp + fn, 1)
        fa_rate    = fp / max(fp + tn, 1)
        print(f"    Catch rate          : {catch_rate:.1%}")
        print(f"    False alarm rate    : {fa_rate:.2%}")


def main():
    #  Train 
    detector = FraudDetector(model_dir=MODEL_DIR, docs_dir=DOCS_DIR)
    results  = detector.fit_all(DATA_PATH)

    #  Summary 
    print_results(results)

    #  SHAP explainability
    run_shap(DATA_PATH, model_name=BEST_MODEL,
             model_dir=MODEL_DIR, docs_dir=DOCS_DIR)


if __name__ == "__main__":
    main()