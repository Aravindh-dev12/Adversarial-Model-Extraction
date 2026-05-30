"""Query strategies for active model extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import pairwise_distances

from .metrics import entropy, margin_uncertainty, normalize_probabilities


@dataclass(frozen=True)
class QueryStrategy:
    """Metadata for a query-selection strategy."""

    name: str
    description: str


_STRATEGIES = {
    "random": QueryStrategy("random", "Uniform random sampling from the candidate pool."),
    "entropy": QueryStrategy("entropy", "Query samples where the student is most uncertain."),
    "margin": QueryStrategy("margin", "Query samples with the smallest top-2 class margin."),
    "disagreement": QueryStrategy("disagreement", "Query samples where surrogate models disagree."),
    "kcenter": QueryStrategy("kcenter", "Diversity sampling with greedy k-center coverage."),
    "hybrid": QueryStrategy("hybrid", "Blend uncertainty, disagreement, and diversity."),
}


def list_strategies() -> list[str]:
    """Return supported strategy names."""
    return list(_STRATEGIES)


def get_strategy(name: str) -> QueryStrategy:
    """Return strategy metadata by name."""
    key = name.lower().strip()
    if key not in _STRATEGIES:
        raise ValueError(f"Unknown query strategy '{name}'. Choose from: {', '.join(list_strategies())}")
    return _STRATEGIES[key]


def _standardize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    spread = float(scores.max() - scores.min())
    if spread <= 1e-12:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - scores.min()) / spread


def _top_k(scores: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    k = int(min(k, scores.shape[0]))
    if k <= 0:
        return np.array([], dtype=np.int64)

    jitter = rng.uniform(0.0, 1e-9, size=scores.shape[0])
    ranked = np.argsort(-(scores + jitter), kind="mergesort")
    return ranked[:k].astype(np.int64)


def _random(k: int, n_candidates: int, rng: np.random.Generator) -> np.ndarray:
    k = int(min(k, n_candidates))
    if k <= 0:
        return np.array([], dtype=np.int64)
    return rng.choice(n_candidates, size=k, replace=False).astype(np.int64)


def _disagreement_score(ensemble_probs: list[np.ndarray] | np.ndarray | None) -> np.ndarray:
    if ensemble_probs is None:
        raise ValueError("disagreement strategy requires ensemble_probs")

    stack = np.asarray(ensemble_probs, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError("ensemble_probs must have shape [models, samples, classes]")

    normalized = np.stack([normalize_probabilities(member) for member in stack], axis=0)
    return np.mean(np.var(normalized, axis=0), axis=1)


def _kcenter(
    candidate_features,
    labeled_features,
    k: int,
    rng: np.random.Generator,
    metric: str,
) -> np.ndarray:
    n_candidates = candidate_features.shape[0]
    k = int(min(k, n_candidates))
    if k <= 0:
        return np.array([], dtype=np.int64)

    if labeled_features is None or labeled_features.shape[0] == 0:
        selected = [int(rng.integers(0, n_candidates))]
        min_dist = pairwise_distances(
            candidate_features,
            candidate_features[selected],
            metric=metric,
        ).reshape(-1)
    else:
        min_dist = pairwise_distances(candidate_features, labeled_features, metric=metric).min(axis=1)
        selected = []

    selected_mask = np.zeros(n_candidates, dtype=bool)
    for idx in selected:
        selected_mask[idx] = True

    while len(selected) < k:
        masked = np.where(selected_mask, -np.inf, min_dist)
        next_idx = int(np.argmax(masked))
        selected.append(next_idx)
        selected_mask[next_idx] = True

        new_dist = pairwise_distances(
            candidate_features,
            candidate_features[next_idx : next_idx + 1],
            metric=metric,
        ).reshape(-1)
        min_dist = np.minimum(min_dist, new_dist)

    return np.asarray(selected, dtype=np.int64)


def select_queries(
    strategy: str,
    k: int,
    *,
    rng: np.random.Generator,
    student_probs: np.ndarray | None = None,
    ensemble_probs: list[np.ndarray] | np.ndarray | None = None,
    candidate_features=None,
    labeled_features=None,
    distance_metric: str = "euclidean",
) -> np.ndarray:
    """Select local candidate positions to query next."""
    name = get_strategy(strategy).name

    n_candidates = None
    if student_probs is not None:
        n_candidates = np.asarray(student_probs).shape[0]
    elif candidate_features is not None:
        n_candidates = candidate_features.shape[0]
    elif ensemble_probs is not None:
        n_candidates = np.asarray(ensemble_probs).shape[1]

    if n_candidates is None:
        raise ValueError("At least one of student_probs, ensemble_probs, or candidate_features is required")

    if name == "random":
        return _random(k, n_candidates, rng)

    if name == "entropy":
        if student_probs is None:
            raise ValueError("entropy strategy requires student_probs")
        return _top_k(entropy(student_probs), k, rng)

    if name == "margin":
        if student_probs is None:
            raise ValueError("margin strategy requires student_probs")
        return _top_k(margin_uncertainty(student_probs), k, rng)

    if name == "disagreement":
        return _top_k(_disagreement_score(ensemble_probs), k, rng)

    if name == "kcenter":
        if candidate_features is None:
            raise ValueError("kcenter strategy requires candidate_features")
        return _kcenter(candidate_features, labeled_features, k, rng, distance_metric)

    if name == "hybrid":
        if student_probs is None or ensemble_probs is None or candidate_features is None:
            raise ValueError("hybrid strategy requires student_probs, ensemble_probs, and candidate_features")

        uncertainty = _standardize(entropy(student_probs))
        disagreement = _standardize(_disagreement_score(ensemble_probs))

        if labeled_features is not None and labeled_features.shape[0] > 0:
            diversity = pairwise_distances(candidate_features, labeled_features, metric=distance_metric).min(axis=1)
            diversity = _standardize(diversity)
        else:
            diversity = np.zeros(candidate_features.shape[0], dtype=np.float64)

        scores = (0.45 * uncertainty) + (0.35 * disagreement) + (0.20 * diversity)
        return _top_k(scores, k, rng)

    raise AssertionError(f"Unhandled strategy: {name}")
