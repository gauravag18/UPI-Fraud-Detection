from src.transaction.preprocess import preprocess_pipeline
from src.transaction.train_random_forest import train_random_forest

X, y = preprocess_pipeline("transaction_data/upi_transactions_2024.csv")

model, X_test, y_test = train_random_forest(X, y)