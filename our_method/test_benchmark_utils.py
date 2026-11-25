import csv
import tempfile
import unittest
from pathlib import Path

from benchmark_utils import append_result_csv, load_tasks, score_hypotheses


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


if __name__ == "__main__":
    unittest.main()
