from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def score(y_true, y_pred, labels):
    n = list(range(len(labels)))
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, labels=n, target_names=labels,
                                        output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=n).tolist(),
    }
