import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Build confusion matrix
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    
    # Compute TP, FP, FN for each class
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    support = np.sum(cm, axis=1)
    
    # Accuracy (same for all modes)
    accuracy = np.sum(TP) / np.sum(cm)
    
    def safe_div(a, b):
        return a / b if b != 0 else 0.0
    
    # Per-class metrics
    precision_c = [safe_div(TP[i], TP[i] + FP[i]) for i in range(n_classes)]
    recall_c = [safe_div(TP[i], TP[i] + FN[i]) for i in range(n_classes)]
    f1_c = [safe_div(2 * precision_c[i] * recall_c[i], precision_c[i] + recall_c[i])
            for i in range(n_classes)]
    
    if average == "micro":
        TP_sum = np.sum(TP)
        FP_sum = np.sum(FP)
        FN_sum = np.sum(FN)
        
        precision = safe_div(TP_sum, TP_sum + FP_sum)
        recall = safe_div(TP_sum, TP_sum + FN_sum)
        f1 = safe_div(2 * precision * recall, precision + recall)
    
    elif average == "macro":
        precision = np.mean(precision_c)
        recall = np.mean(recall_c)
        f1 = np.mean(f1_c)
    
    elif average == "weighted":
        weights = support / np.sum(support)
        precision = np.sum(weights * precision_c)
        recall = np.sum(weights * recall_c)
        f1 = np.sum(weights * f1_c)
    
    elif average == "binary":
        i = class_to_idx[pos_label]
        precision = precision_c[i]
        recall = recall_c[i]
        f1 = f1_c[i]
    
    else:
        raise ValueError("Invalid averaging method")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# Example usage
result = classification_metrics([0,1,2,2], [0,1,0,2], average="micro")
print(result)