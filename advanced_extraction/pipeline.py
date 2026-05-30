"""CPU-friendly advanced extraction simulation.

The simulation trains a hidden teacher model, exposes only black-box
probability queries to an attacker, and measures how efficiently different
active query strategies reproduce the teacher's top-1 behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import (
    expected_calibration_error,
    fidelity_score,
    kl_divergence,
    normalize_probabilities,
)
from .strategies import list_strategies, select_queries


@dataclass
class ExtractionConfig:
    """Configuration for the advanced extraction simulator."""

    strategy: str = "hybrid"
    n_samples: int = 2400
    n_features: int = 28
    n_classes: int = 3
    query_budget: int = 320
    initial_size: int = 36
    step_size: int = 32
    candidate_size: int = 420
    random_state: int = 42
    teacher_trees: int = 180


@dataclass
class ExtractionResult:
    """Result object returned by the simulator."""

    config: dict[str, Any]
    history: list[dict[str, Any]]
    final_fidelity: float
    final_kl: float
    final_ece: float
    query_count: int
    teacher_task_accuracy: float
    student_task_accuracy: float


def _make_dataset(config: ExtractionConfig):
    informative = max(config.n_classes, int(config.n_features * 0.55))
    redundant = max(0, int(config.n_features * 0.20))
    redundant = min(redundant, config.n_features - informative)

    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=informative,
        n_redundant=redundant,
        n_repeated=0,
        n_classes=config.n_classes,
        n_clusters_per_class=1,
        class_sep=1.35,
        flip_y=0.025,
        random_state=config.random_state,
    )

    X_teacher, X_public, y_teacher, y_public = train_test_split(
        X,
        y,
        test_size=0.62,
        stratify=y,
        random_state=config.random_state,
    )
    X_pool, X_val, y_pool, y_val = train_test_split(
        X_public,
        y_public,
        test_size=0.30,
        stratify=y_public,
        random_state=config.random_state + 1,
    )
    return X_teacher, y_teacher, X_pool, y_pool, X_val, y_val


def _make_teacher(config: ExtractionConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.teacher_trees,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=config.random_state,
        n_jobs=1,
    )


def _make_soft_student(seed: int, alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("reg", MultiOutputRegressor(Ridge(alpha=alpha, random_state=seed))),
        ]
    )


def _make_logistic_student(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.6,
                    class_weight="balanced",
                    max_iter=900,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _make_sgd_student(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=0.0006,
                    max_iter=1400,
                    tol=1e-4,
                    random_state=seed,
                ),
            ),
        ]
    )


def _make_tree_student(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=90,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def _balanced_seed_indices(labels: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(labels)
    selected: list[int] = []

    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        if positions.size:
            selected.append(int(rng.choice(positions)))
        if len(selected) >= size:
            return np.asarray(selected, dtype=np.int64)

    remaining = np.setdiff1d(np.arange(labels.shape[0]), np.asarray(selected, dtype=np.int64))
    needed = max(0, min(size, labels.shape[0]) - len(selected))
    if needed:
        selected.extend(rng.choice(remaining, size=needed, replace=False).astype(int).tolist())

    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def _fit_students(
    X_train: np.ndarray,
    teacher_probs: np.ndarray,
    random_state: int,
):
    teacher_top1 = normalize_probabilities(teacher_probs).argmax(axis=1)
    soft_a = _make_soft_student(random_state, alpha=0.9)
    soft_b = _make_soft_student(random_state + 17, alpha=3.0)
    logreg = _make_logistic_student(random_state + 29)
    sgd = _make_sgd_student(random_state + 43)
    trees = _make_tree_student(random_state + 59)

    soft_a.fit(X_train, teacher_probs)
    soft_b.fit(X_train, teacher_probs)
    logreg.fit(X_train, teacher_top1)
    sgd.fit(X_train, teacher_top1)
    trees.fit(X_train, teacher_top1)
    return soft_a, soft_b, logreg, sgd, trees


def _classifier_probs(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.full((X.shape[0], n_classes), 1e-12, dtype=np.float64)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = model.named_steps["clf"].classes_
    for source_idx, class_id in enumerate(classes):
        out[:, int(class_id)] = raw[:, source_idx]
    return normalize_probabilities(out)


def _student_ensemble(students, X: np.ndarray, n_classes: int) -> list[np.ndarray]:
    soft_a, soft_b, logreg, sgd, trees = students
    return [
        normalize_probabilities(soft_a.predict(X)),
        normalize_probabilities(soft_b.predict(X)),
        _classifier_probs(logreg, X, n_classes),
        _classifier_probs(sgd, X, n_classes),
        _classifier_probs(trees, X, n_classes),
    ]


def _record_metrics(
    *,
    round_id: int,
    strategy: str,
    query_count: int,
    train_size: int,
    pool_left: int,
    teacher_probs: np.ndarray,
    student_probs: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    teacher_top1 = normalize_probabilities(teacher_probs).argmax(axis=1)
    student_top1 = normalize_probabilities(student_probs).argmax(axis=1)
    return {
        "round": round_id,
        "strategy": strategy,
        "queries": int(query_count),
        "train_size": int(train_size),
        "pool_left": int(pool_left),
        "fidelity": fidelity_score(teacher_probs, student_probs),
        "kl_teacher_student": kl_divergence(teacher_probs, student_probs),
        "ece": expected_calibration_error(teacher_probs, student_probs),
        "teacher_task_accuracy": float(accuracy_score(y_val, teacher_top1)),
        "student_task_accuracy": float(accuracy_score(y_val, student_top1)),
    }


def run_extraction_simulation(config: ExtractionConfig | None = None) -> ExtractionResult:
    """Run a black-box active model extraction simulation."""
    config = config or ExtractionConfig()
    if config.n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if config.initial_size < config.n_classes:
        raise ValueError("initial_size must be at least n_classes so students see every class")
    if config.query_budget < config.initial_size:
        raise ValueError("query_budget must be greater than or equal to initial_size")

    rng = np.random.default_rng(config.random_state)
    X_teacher, y_teacher, X_pool, _y_pool, X_val, y_val = _make_dataset(config)

    teacher = _make_teacher(config)
    teacher.fit(X_teacher, y_teacher)

    pool_teacher_probs = normalize_probabilities(teacher.predict_proba(X_pool))
    val_teacher_probs = normalize_probabilities(teacher.predict_proba(X_val))
    pool_teacher_top1 = pool_teacher_probs.argmax(axis=1)

    labeled = _balanced_seed_indices(pool_teacher_top1, config.initial_size, rng)
    unlabeled = np.setdiff1d(np.arange(X_pool.shape[0]), labeled)
    query_count = int(labeled.shape[0])
    history: list[dict[str, Any]] = []
    round_id = 0
    final_student_probs = None

    while True:
        X_labeled = X_pool[labeled]
        y_labeled_probs = pool_teacher_probs[labeled]
        students = _fit_students(X_labeled, y_labeled_probs, config.random_state + round_id)

        final_student_probs = normalize_probabilities(students[0].predict(X_val))
        history.append(
            _record_metrics(
                round_id=round_id,
                strategy=config.strategy,
                query_count=query_count,
                train_size=labeled.shape[0],
                pool_left=unlabeled.shape[0],
                teacher_probs=val_teacher_probs,
                student_probs=final_student_probs,
                y_val=y_val,
            )
        )

        if query_count >= config.query_budget or unlabeled.size == 0:
            break

        candidate_count = int(min(config.candidate_size, unlabeled.shape[0]))
        candidate_positions = rng.choice(unlabeled, size=candidate_count, replace=False)
        X_candidate = X_pool[candidate_positions]
        candidate_student_probs = normalize_probabilities(students[0].predict(X_candidate))
        candidate_ensemble = _student_ensemble(students, X_candidate, config.n_classes)

        step = int(min(config.step_size, config.query_budget - query_count, candidate_positions.shape[0]))
        selected_local = select_queries(
            config.strategy,
            step,
            rng=rng,
            student_probs=candidate_student_probs,
            ensemble_probs=candidate_ensemble,
            candidate_features=X_candidate,
            labeled_features=X_labeled,
            distance_metric="euclidean",
        )

        selected = candidate_positions[selected_local]
        labeled = np.concatenate([labeled, selected])
        unlabeled = np.setdiff1d(unlabeled, selected)
        query_count += int(selected.shape[0])
        round_id += 1

    assert final_student_probs is not None
    final = history[-1]
    return ExtractionResult(
        config=asdict(config),
        history=history,
        final_fidelity=float(final["fidelity"]),
        final_kl=float(final["kl_teacher_student"]),
        final_ece=float(final["ece"]),
        query_count=query_count,
        teacher_task_accuracy=float(final["teacher_task_accuracy"]),
        student_task_accuracy=float(final["student_task_accuracy"]),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run advanced active model extraction simulation.")
    parser.add_argument("--strategy", choices=list_strategies(), default="hybrid")
    parser.add_argument("--n-samples", type=int, default=2400)
    parser.add_argument("--n-features", type=int, default=28)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--query-budget", type=int, default=320)
    parser.add_argument("--initial-size", type=int, default=36)
    parser.add_argument("--step-size", type=int, default=32)
    parser.add_argument("--candidate-size", type=int, default=420)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/advanced_extraction_result.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ExtractionConfig(
        strategy=args.strategy,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        query_budget=args.query_budget,
        initial_size=args.initial_size,
        step_size=args.step_size,
        candidate_size=args.candidate_size,
        random_state=args.random_state,
    )
    result = run_extraction_simulation(config)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)

    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
