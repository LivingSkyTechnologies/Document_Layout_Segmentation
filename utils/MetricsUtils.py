# Mostly borrowed from https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/utils/utils.py

import numpy as np
from sklearn.metrics import precision_score, recall_score, \
    confusion_matrix, accuracy_score, f1_score


def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT, 
    # it returns NAN because of divide by zero
    # Ignore totals of zero
    accuracies = {}
    for i in range(len(total)):
        if total[i] == 0:
            continue
        else:
            accuracies[i] = count[i] / total[i]

    return accuracies

def compute_mean_iou(pred, label, num_classes):

    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    iou = np.zeros((num_classes,))
    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I = float(np.sum(np.logical_and(label_i, pred_i)))
        U = float(np.sum(np.logical_or(label_i, pred_i)))
        
        iou[int(val)] = np.mean(I/U)

    return iou

def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()
    
    # Convert don't cares to background class and make pred match ground truth
    flat_pred = np.where(flat_label == 255.0, 0.0, flat_pred)
    flat_label = np.where(flat_label == 255.0, 0.0, flat_label)
    
    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging, zero_division=1)
    rec = recall_score(flat_pred, flat_label, average=score_averaging, zero_division=1)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging, zero_division=1)

    iou = compute_mean_iou(flat_pred, flat_label, num_classes)

    return global_accuracy, class_accuracies, prec, rec, f1, iou


