import unittest

from boolean_genetics import (
    Observation,
    evaluate_expression,
    expression_mechanistic_key,
    sample_hypotheses,
)


OBSERVATIONS = (
    Observation({"A": False, "B": False}, False),
    Observation({"A": False, "B": True}, False),
    Observation({"A": True, "B": False}, False),
    Observation({"A": True, "B": True}, True),
)


class SamplingTests(unittest.TestCase):
    def test_semantically_equivalent_candidates_are_rejected(self):
        responses = iter(("A AND B", "B and A", "A"))
        prompts = []
        temperatures = []

        def complete(prompt, temperature):
            prompts.append(prompt)
            temperatures.append(temperature)
            return next(responses)

        partial_observations = (Observation({"A": True, "B": True}, True),)
        result = sample_hypotheses(partial_observations, complete, target_count=2, max_attempts=3)

        self.assertEqual(result.hypotheses, ("A AND B", "A"))
        self.assertEqual(result.rejection_counts["duplicate"], 1)
        self.assertEqual(temperatures, [1.0, 0.6, 0.6])
        self.assertIn("duplicates an accepted hypothesis under semantic deduplication", prompts[2])

    def test_invalid_and_inconsistent_candidates_receive_feedback(self):
        responses = iter(("A xor B", "A OR B", "A AND B"))
        prompts = []

        def complete(prompt, _temperature):
            prompts.append(prompt)
            return next(responses)

        result = sample_hypotheses(OBSERVATIONS, complete, target_count=1, max_attempts=3)

        self.assertEqual(result.hypotheses, ("A AND B",))
        self.assertEqual(result.proposals, ("A xor B", "A OR B", "A AND B"))
        self.assertEqual(result.raw_responses, ("A xor B", "A OR B", "A AND B"))
        self.assertEqual(result.rejection_counts, {"invalid": 1, "inconsistent": 1, "duplicate": 0})
        self.assertIn("Previous rejection:", prompts[1])
        self.assertIn("does not match all observed outputs", prompts[2])

    def test_attempt_limit_reports_partial_result(self):
        result = sample_hypotheses(OBSERVATIONS, lambda *_: "A", target_count=2, max_attempts=1)

        self.assertFalse(result.reached_target)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.hypotheses, ())

    def test_xor_and_nor_are_supported(self):
        self.assertTrue(evaluate_expression("x XOR y", {"x": True, "y": False}))
        self.assertFalse(evaluate_expression("NOR(x, y)", {"x": False, "y": True}))
        self.assertEqual(
            expression_mechanistic_key("NOT (x OR y)"),
            expression_mechanistic_key("NOR(x, y)"),
        )

    def test_operator_and_depth_constraints_are_enforced(self):
        responses = iter(("x XOR y", "x AND y"))

        result = sample_hypotheses(
            (Observation({"x": True, "y": True}, True),),
            lambda *_: next(responses),
            target_count=1,
            max_attempts=2,
            allowed_operators=("AND", "OR", "NOT"),
            max_depth=1,
            allow_constants=False,
        )

        self.assertEqual(result.hypotheses, ("x AND y",))
        self.assertEqual(result.rejection_counts["invalid"], 1)

    def test_mechanistic_dedup_keeps_structurally_distinct_equivalent_forms(self):
        responses = iter(("x AND y", "NOT (NOT x OR NOT y)"))
        observations = (Observation({"x": True, "y": True}, True),)

        result = sample_hypotheses(
            observations,
            lambda *_: next(responses),
            target_count=2,
            max_attempts=2,
            allowed_operators=("AND", "OR", "NOT"),
            deduplication="mechanistic",
        )

        self.assertEqual(len(result.hypotheses), 2)
        self.assertEqual(result.rejection_counts["duplicate"], 0)

    def test_mechanistic_dedup_rejects_canonical_duplicates(self):
        responses = iter(("x AND y", "y AND x"))
        observations = (Observation({"x": True, "y": True}, True),)

        result = sample_hypotheses(
            observations,
            lambda *_: next(responses),
            target_count=2,
            max_attempts=2,
            allowed_operators=("AND", "OR", "NOT"),
            deduplication="mechanistic",
        )

        self.assertEqual(result.hypotheses, ("x AND y",))
        self.assertEqual(result.rejection_counts["duplicate"], 1)

    def test_fixed_proposal_budget_records_duplicate_outputs(self):
        prompts = []

        def complete(prompt, _temperature):
            prompts.append(prompt)
            return "A"

        result = sample_hypotheses(
            (Observation({"A": True}, True),),
            complete,
            target_count=1,
            max_attempts=3,
            start_temperature=0.7,
            end_temperature=0.7,
            deduplication="none",
            include_history=False,
            stop_on_target=False,
        )

        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.proposals, ("A", "A", "A"))
        self.assertEqual(result.raw_responses, ("A", "A", "A"))
        self.assertEqual(result.hypotheses, ("A", "A", "A"))
        self.assertEqual(len(set(prompts)), 1)


if __name__ == "__main__":
    unittest.main()
