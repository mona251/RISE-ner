import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score


def compute_metrics(p, metric, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def compute_inference_metrics(label, pred, metric, confusion_matrix, label_list, label2id):
    
    true_predictions = [label_list[p] for (p, l) in zip(pred, label) if l != -100]
    true_labels = [label_list[l] for (p, l) in zip(pred, label) if l != -100]

    confusion_pred = [label2id[i] for i in true_predictions]
    confusion_label = [label2id[i] for i in true_labels]
    results = metric.compute(predictions=[true_predictions], references=[true_labels])
    confusion = confusion_matrix.compute(predictions=confusion_pred, references=confusion_label)
    kappa = cohen_kappa_score(true_predictions, true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }, confusion, results, kappa
