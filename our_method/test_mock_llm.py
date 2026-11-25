import unittest

from boolean_genetics import Observation, sample_hypotheses
from mock_llm import MockLLM, mock_baseline_response


class MockLLMTests(unittest.TestCase):
    def test_bad_formats_inconsistency_and_duplicate_are_rejected_then_recovered(self):
        observations = (Observation({"x": True, "y": True}, True),)
        result = sample_hypotheses(
            observations,
            MockLLM(),
            target_count=2,
            max_attempts=10,
            allowed_operators=("AND", "OR", "NOT"),
            max_depth=1,
            allow_constants=False,
        )

        self.assertTrue(result.reached_target)
        self.assertEqual(len(result.hypotheses), 2)
        self.assertEqual(result.rejection_counts, {
            "invalid": 1,
            "inconsistent": 1,
            "duplicate": 1,
        })

    def test_baseline_mock_returns_formatted_legal_candidates(self):
        response = mock_baseline_response(
            (Observation({"x": True, "y": True}, True),),
            ("x", "y"),
            2,
            ("AND", "OR", "NOT"),
            1,
        )

        self.assertIn("1. Expression:", response)
        self.assertEqual(len(response.splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
