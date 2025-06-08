from sklearn.metrics import classification_report, accuracy_score, f1_score

def evaluate_classification(y_true, y_pred, average='binary'):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred, average=average))
    print("Full Report:\n", classification_report(y_true, y_pred))
