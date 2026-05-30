import unittest

import numpy as np

from advanced_extraction import ExtractionConfig, run_extraction_simulation, select_queries
from advanced_extraction.metrics import normalize_probabilities


class AdvancedExtractionTests(unittest.TestCase):
    def test_normalize_probabilities_handles_invalid_rows(self):
        probs = normalize_probabilities(np.array([[0.0, 0.0, 0.0], [2.0, -1.0, np.nan]]))

        self.assertEqual(probs.shape, (2, 3))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2))
        self.assertTrue(np.all(probs > 0.0))

    def test_entropy_strategy_prefers_uncertain_predictions(self):
        rng = np.random.default_rng(7)
        student_probs = np.array(
            [
                [0.96, 0.02, 0.02],
                [0.34, 0.33, 0.33],
                [0.80, 0.10, 0.10],
            ]
        )

        selected = select_queries("entropy", 1, rng=rng, student_probs=student_probs)

        self.assertEqual(selected.tolist(), [1])

    def test_kcenter_returns_unique_candidates(self):
        rng = np.random.default_rng(11)
        candidates = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
        labeled = np.array([[0.0]])

        selected = select_queries(
            "kcenter",
            3,
            rng=rng,
            candidate_features=candidates,
            labeled_features=labeled,
        )

        self.assertEqual(len(selected), len(set(selected.tolist())))
        self.assertEqual(len(selected), 3)

    def test_simulation_runs_with_small_budget(self):
        config = ExtractionConfig(
            strategy="hybrid",
            n_samples=520,
            n_features=12,
            n_classes=3,
            query_budget=36,
            initial_size=12,
            step_size=8,
            candidate_size=40,
            random_state=3,
            teacher_trees=30,
        )

        result = run_extraction_simulation(config)

        self.assertEqual(result.query_count, 36)
        self.assertGreaterEqual(result.final_fidelity, 0.0)
        self.assertLessEqual(result.final_fidelity, 1.0)
        self.assertGreaterEqual(len(result.history), 2)


if __name__ == "__main__":
    unittest.main()
