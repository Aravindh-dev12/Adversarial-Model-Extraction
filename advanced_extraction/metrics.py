"""Metrics for model extraction experiments."""

from __future__ import annotations

import numpy as np


def normalize_probabilities(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clip and row-normalize model outputs into a probability simplex."""
    probs = np.asarray(values, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, eps, None)
    row_sum = probs.sum(axis=1, keepdims=True)

    invalid = row_sum.squeeze(axis=1) <= eps
    if np.any(invalid):
        probs[invalid] = 1.0
        row_sum = probs.sum(axis=1, keepdims=True)

    return probs / row_sum


def entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return per-row Shannon entropy."""
    p = normalize_probabilities(probs, eps=eps)
    return -np.sum(p * np.log(p + eps), axis=1)


def margin_uncertainty(probs: np.ndarray) -> np.ndarray:
    """Return uncertainty score based on the top-2 probability margin."""
    p = normalize_probabilities(probs)
    if p.shape[1] == 1:
        return np.ones(p.shape[0], dtype=np.float64)

    top2 = np.partition(p, kth=-2, axis=1)[:, -2:]
    top2.sort(axis=1)
    margin = top2[:, 1] - top2[:, 0]
    return 1.0 - margin


def kl_divergence(
    teacher_probs: np.ndarray,
    student_probs: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Average KL(teacher || student)."""
    teacher = normalize_probabilities(teacher_probs, eps=eps)
    student = normalize_probabilities(student_probs, eps=eps)
    return float(np.mean(np.sum(teacher * (np.log(teacher + eps) - np.log(student + eps)), axis=1)))


def fidelity_score(teacher_probs: np.ndarray, student_probs: np.ndarray) -> float:
    """Top-1 agreement between teacher and student distributions."""
    teacher_top1 = normalize_probabilities(teacher_probs).argmax(axis=1)
    student_top1 = normalize_probabilities(student_probs).argmax(axis=1)
    return float(np.mean(teacher_top1 == student_top1))


def expected_calibration_error(
    teacher_probs: np.ndarray,
    student_probs: np.ndarray,
    bins: int = 10,
) -> float:
    """ECE using teacher top-1 as the reference label source."""
    teacher = normalize_probabilities(teacher_probs)
    student = normalize_probabilities(student_probs)
    teacher_top1 = teacher.argmax(axis=1)
    student_top1 = student.argmax(axis=1)
    confidence = student.max(axis=1)
    correct = (student_top1 == teacher_top1).astype(np.float64)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not np.any(mask):
            continue
        weight = float(np.mean(mask))
        ece += weight * abs(float(np.mean(confidence[mask])) - float(np.mean(correct[mask])))
    return float(ece)
