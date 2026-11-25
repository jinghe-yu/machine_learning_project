import csv
import tempfile
import unittest
from pathlib import Path

from benchmark_utils import (
    append_result_csv,
    load_tasks,
    score_hypotheses,
    task_target_count,
)
from boolean_genetics import expression_mechanistic_key


class BenchmarkMetricTests(unittest.TestCase):
    def test_official_ground_truths_recover_on_generated_dataset(self):
        project_dir = Path(__file__).parent
        metadata, tasks = load_tasks(project_dir / "data" / "boolean_dataset.json")

        for task in tasks:
            formulas = [item["formula"] for item in task["ground_truth_expressions"]]
            result = score_hypotheses(task, formulas, metadata)

            with self.subTest(observation_set=task["observation_set_id"]):
                self.assertEqual(
                    result["n_recovered_structural"], result["ground_truth_size"]
                )
                self.assertEqual(
                    result["n_recovered_semantic"],
                    result["semantic_ground_truth_size"],
                )

    def test_append_result_csv_writes_header_once_and_preserves_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "results.csv"
            append_result_csv(output, {"observation_set_id": "first", "attempts": 1})
            append_result_csv(output, {"observation_set_id": "second", "attempts": 2})

            with output.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(
            [(row["observation_set_id"], row["attempts"]) for row in rows],
            [("first", "1"), ("second", "2")],
        )

    def test_append_result_csv_rejects_an_old_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "old.csv"
            output.write_text("method,observation_set_id\nbaseline,first\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                append_result_csv(output, {"observation_set_id": "second"})

    def test_metrics_use_mechanistic_ground_truth_and_proposal_denominators(self):
        simple = "x AND y"
        complex_expression = "NOT (NOT x OR NOT y) AND (x OR y)"
        task = {
            "observation_set_id": "metric-test",
            "n_observations": 1,
            "observations": [{"inputs": {"x": True, "y": True}, "output": True}],
            "ground_truth_expressions": [
                {
                    "formula": formula,
                    "mechanistic_key": repr(expression_mechanistic_key(formula)),
                }
                for formula in (simple, complex_expression)
            ],
        }
        metadata = {
            "variables": ["x", "y"],
            "operators": ["AND", "OR", "NOT"],
            "max_depth": 4,
        }
        candidates = [simple, simple, complex_expression, "NOT x", "z", None]

        result = score_hypotheses(task, candidates, metadata)

        self.assertEqual(task_target_count(task, metadata["variables"]), 2)
        self.assertEqual(result["n_proposals"], 6)
        self.assertAlmostEqual(result["validity_rate"], 3 / 6)
        self.assertAlmostEqual(result["novelty_rate"], 3 / 6)
        self.assertEqual(result["recovery_rate"], 1.0)
        self.assertEqual(result["simple_recovery_rate"], 1.0)
        self.assertEqual(result["complex_recovery_rate"], 1.0)
        self.assertEqual(result["n_recovered_semantic"], 1)


if __name__ == "__main__":
    unittest.main()
