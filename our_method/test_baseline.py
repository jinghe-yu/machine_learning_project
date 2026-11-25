import csv
import tempfile
import unittest
from pathlib import Path

from benchmark_utils import load_completed_task_ids


class BaselineResumeTests(unittest.TestCase):
    def test_load_completed_task_ids_filters_by_method(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results.csv"
            with output.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file, fieldnames=["method", "observation_set_id"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"method": "baseline", "observation_set_id": "done"},
                        {"method": "ours", "observation_set_id": "other-method"},
                    ]
                )

            completed = load_completed_task_ids(output, "baseline")

        self.assertEqual(completed, {"done"})


if __name__ == "__main__":
    unittest.main()
