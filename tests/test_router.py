"""End-to-end router tests using explicitly synthetic temporary fixtures only."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from study.common import ROOT, context_hash, load_config, write_jsonl
from study.data import load_data
from study.prompts import build_grid
from study.results import load_scored
from study.router import evaluate_test, features, route_question, train


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.manifest, cls.records = load_data(cls.config, ROOT / "data")
        cls.gold = {(row["problem_id"], row["language"]): row["gold"] for row in cls.records}

    def fixture_rows(self, phase):
        rows = []
        for job in build_grid(self.config, self.manifest, self.records, phase, context_hash(self.config, self.manifest)):
            target_budget = [0, 2, 4][int(job["problem_id"].split("-")[1]) % 3]
            success = job["budget"] >= target_budget and job["budget"] != 16
            gold = self.gold[job["problem_id"], job["language"]]
            answer = gold if success else "99999999999999999"
            rows.append({**job, "backend": "transformers", "status": "complete",
                         "response": "\n".join(f"{step + 1}. Synthetic test fixture." for step in range(job["budget"])) + f"\nFINAL_ANSWER: {answer}",
                         "output_tokens": 10 + 5 * job["budget"], "truncated": False,
                         "latency_seconds": 0.0})
        return rows

    def test_features_are_question_only_and_unicode_aware(self):
        value = features("There are ১২ eggs and ౩ boxes. How many?", "bn")
        self.assertEqual(value["quantity_count"], 2)
        self.assertEqual(value["language"], "bn")
        self.assertEqual(set(value), {"character_length", "sentence_count", "quantity_count", "language"})
        self.assertNotIn("language", features("Question?", "te", include_language=False))

    def test_incomplete_development_cannot_unlock_test(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "4,600"):
                train(self.config, self.manifest, self.records, [], Path(directory))

    def test_grouped_fit_freeze_cached_test_and_tamper_detection(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-router-test-") as directory:
            folder = Path(directory)
            development = self.fixture_rows("development")
            write_jsonl(folder / "raw/synthetic.jsonl", development)
            scored = load_scored(self.config, self.manifest, self.records, folder)
            metrics = train(self.config, self.manifest, self.records, scored, folder)
            self.assertEqual(metrics["phase"], "grouped_development_cross_validation")
            self.assertEqual(len(metrics["summary"]), 10)
            self.assertEqual(len(metrics["language_summary"]), 30)
            self.assertEqual({row["questions"] for row in metrics["summary"]}, {300})
            route = route_question(self.config, self.manifest, folder, "qwen3-1.7b", "bn", "There are ১২ eggs. How many?")
            self.assertIn(route["budget"], self.config["budgets"])
            with self.assertRaisesRegex(ValueError, "already frozen"):
                train(self.config, self.manifest, self.records, scored, folder)
            test_rows = self.fixture_rows("test")
            write_jsonl(folder / "raw/synthetic.jsonl", development + test_rows)
            test_metrics = evaluate_test(self.config, self.manifest, self.records,
                                         load_scored(self.config, self.manifest, self.records, folder), folder)
            self.assertEqual(test_metrics["phase"], "held_out_test")
            self.assertEqual({row["questions"] for row in test_metrics["summary"]}, {450})
            # Held-out outcomes cannot change already-fitted decisions.
            self.assertEqual(route, route_question(self.config, self.manifest, folder, "qwen3-1.7b", "bn", "There are ১২ eggs. How many?"))
            development[0]["response"] = "FINAL_ANSWER: 99999"
            write_jsonl(folder / "raw/synthetic.jsonl", development + test_rows)
            with self.assertRaisesRegex(ValueError, "Development outputs changed"):
                load_scored(self.config, self.manifest, self.records, folder)


if __name__ == "__main__":
    unittest.main()
