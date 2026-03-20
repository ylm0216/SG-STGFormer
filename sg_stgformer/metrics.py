from __future__ import annotations

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rank_true = _rankdata(y_true)
    rank_pred = _rankdata(y_pred)
    if np.std(rank_true) == 0 or np.std(rank_pred) == 0:
        return 0.0
    return float(np.corrcoef(rank_true, rank_pred)[0, 1])


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1_values = []
    for label in range(num_classes):
        true_positive = np.logical_and(y_true == label, y_pred == label).sum()
        false_positive = np.logical_and(y_true != label, y_pred == label).sum()
        false_negative = np.logical_and(y_true == label, y_pred != label).sum()

        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1_values))


def compute_metrics(
    y_score: np.ndarray,
    y_score_pred: np.ndarray,
    y_label: np.ndarray,
    y_label_pred: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float]:
    return {
        "mae": mae(y_score, y_score_pred),
        "rmse": rmse(y_score, y_score_pred),
        "spearman": spearman(y_score, y_score_pred),
        "accuracy": accuracy(y_label, y_label_pred),
        "f1": macro_f1(y_label, y_label_pred, num_classes),
    }
