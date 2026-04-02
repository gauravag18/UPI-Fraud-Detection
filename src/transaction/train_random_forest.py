from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )

    print("Training Random Forest model")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(" Random Forest :")

    print(classification_report(y_test, y_pred, digits=4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

    return model, X_test, y_test