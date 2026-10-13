"""GPU-independent checks for ordering and infrastructure failure records."""
from pathlib import Path
import tempfile
import unittest
from study.common import read_jsonl
from study.inference import execution_order, log_failure


class InferenceTests(unittest.TestCase):
    def test_longest_first_preserves_complete_problem_blocks_and_ids(self):
        rows = [{"model": model, "problem_id": problem, "budget": budget,
                 "experiment_id": f"{model}-{problem}-{budget}"}
                for model in ["small", "large"] for problem in ["first", "second"]
                for budget in [0, 2, 4, 8, 16]]
        ordered = execution_order(rows)
        self.assertEqual(ordered[0]["budget"], 16)
        self.assertEqual({r["problem_id"] for r in ordered[:5]}, {"first"})
        self.assertEqual([r["budget"] for r in ordered[:5]], [16, 8, 4, 2, 0])
        self.assertEqual({r["experiment_id"] for r in ordered}, {r["experiment_id"] for r in rows})
        self.assertEqual(rows[0]["budget"], 0)

    def test_loading_failure_is_logged_and_rethrown(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "synthetic loading failure"):
                with log_failure(folder, "small", "model_loading", config_hash="test-only"):
                    raise RuntimeError("synthetic loading failure")
            rows = read_jsonl(folder / "raw/small.errors.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["stage"], "model_loading")
            self.assertEqual(rows[0]["config_hash"], "test-only")
            self.assertEqual(rows[0]["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
