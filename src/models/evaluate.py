from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_binary_classifier(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y, pred),
        "precision_risk": precision_score(y, pred, zero_division=0),
        "recall_risk": recall_score(y, pred, zero_division=0),
        "f1_risk": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }
