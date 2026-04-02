from src.transaction.preprocess import preprocess_pipeline

X, y = preprocess_pipeline("transaction_data/upi_transactions_2024.csv")

print(X.shape)
print(y.value_counts())
print("Fraud %:", (y.sum() / len(y)) * 100)