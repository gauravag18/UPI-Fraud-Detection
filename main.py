from src.transaction.preprocess import preprocess_pipeline
from src.transaction.train_logistic import train_logistic

X, y = preprocess_pipeline("transaction_data/upi_transactions_2024.csv")

model, X_test, y_test = train_logistic(X, y)