import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def weighted_f1(labels, scores):
    prediction = np.zeros_like(labels)
    ranking = np.argsort(-scores, axis=1)
    for target, order, output in zip(labels, ranking, prediction):
        output[order[:int(target.sum())]] = 1
    return float(f1_score(labels, prediction, average='weighted', zero_division=0))


def evaluate(labels, scores, frequency):
    ranking = np.argsort(-scores, axis=1)
    positive = labels.sum(1)
    valid = positive > 0
    hits = np.take_along_axis(labels, ranking[:, :10], axis=1).sum(1)
    result = {'W-F1': weighted_f1(labels, scores),
              'Recall@10': float((hits[valid] / positive[valid]).mean()),
              'Micro-AUPRC': float(average_precision_score(labels.ravel(), scores.ravel()))}
    observed = labels.sum(0) > 0
    result['Macro-AUPRC'] = float(average_precision_score(labels[:, observed], scores[:, observed], average='macro'))
    for name, mask in [('LE5-F1', frequency <= 5), ('Mid-F1', (frequency >= 6) & (frequency <= 20)),
                       ('High-F1', frequency > 20)]:
        result[name] = weighted_f1(labels[:, mask], scores[:, mask]) if mask.any() else 0.0
    return result