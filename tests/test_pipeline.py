"""Scientific correctness checks; synthetic responses stay in temporary test folders."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from study.analysis import balanced_rows, bootstrap_interval, paired_contrasts
from study.common import ROOT, context_hash, load_config, write_json, write_jsonl
from study.data import load_data
from study.prompts import build_grid, conditions, render_prompt
from study.results import coverage, load_scored, verify_freeze
from study.scoring import extract_answer, normalize_number, score_response


def fixture_response(job: dict, correct: bool = True) -> dict:
    # Clearly synthetic unit-test input, never a model result in the project results directory.
    return {**job, "backend": "transformers", "status": "complete",
            "response": "\n".join(f"{step + 1}. Test-only explanation." for step in range(job["budget"])) +
                        f"\nFINAL_ANSWER: {18 if correct else 17}",
            "output_tokens": 20 + job["budget"] * 10,
            "latency_seconds": 1.0, "truncated": False}


class ScoringTests(unittest.TestCase):
    def test_equivalent_numerical_representations(self):
        for value in ["১৮", "౧౮", "１８", "18.000", "+18", "00018.0"]:
            self.assertEqual(normalize_number(value), "18")
        self.assertEqual(normalize_number("−0.0"), "0")
        self.assertEqual(normalize_number("1,23,456"), "123456")
        self.assertEqual(normalize_number("1,234,567.50"), "1234567.5")
        self.assertEqual(normalize_number("123456789012345678901234567890123456789"), "123456789012345678901234567890123456789")

    def test_ambiguous_answers_are_not_guessed(self):
        for value in ["18 or 19", "18 dollars", "NaN", "Infinity", "1,2", "18/1", "18."]:
            with self.assertRaises(ValueError):
                normalize_number(value)
        self.assertEqual(extract_answer("18 is the answer"), (None, "missing_marker"))
        self.assertEqual(extract_answer("FINAL_ANSWER: 18\nFINAL_ANSWER: 19"), (None, "multiple_markers"))
        self.assertEqual(extract_answer("Work: 99\nFINAL_ANSWER: ১৮"), ("18", None))
        self.assertEqual(extract_answer("FINAL_ANSWER: 18 or 19"), (None, "invalid_number"))

    def test_intermediate_numbers_cannot_override_final_answer(self):
        row = {"response": "1. Intermediate value is 18.\n2. Change it to 17.\nFINAL_ANSWER: 17", "budget": 2, "reasoning_language": "en"}
        result = score_response(row, "18")
        self.assertFalse(result["correct"])
        self.assertEqual(result["actual_steps"], 2)
        self.assertTrue(result["step_compliant"])


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.manifest, cls.records = load_data(cls.config, ROOT / "data")
        cls.current_hash = context_hash(cls.config, cls.manifest)

    def test_shared_ids_and_no_split_leakage(self):
        self.assertEqual(len(self.records), 750)
        development = set(self.manifest["splits"]["development"])
        test = set(self.manifest["splits"]["test"])
        self.assertFalse(development & test)
        for problem_id in development | test:
            rows = [row for row in self.records if row["problem_id"] == problem_id]
            self.assertEqual({row["language"] for row in rows}, {"en", "bn", "te"})
            self.assertEqual(len({row["gold"] for row in rows}), 1)
            self.assertEqual(len({row["split"] for row in rows}), 1)

    def test_exact_grid_and_nonredundant_conditions(self):
        self.assertEqual(sum(len(conditions(language, self.config["budgets"])) for language in self.config["languages"]), 23)
        expected = {"smoke": 460, "pilot": 2300, "development": 4600, "test": 6900}
        for phase, count in expected.items():
            grid = build_grid(self.config, self.manifest, self.records, phase, self.current_hash)
            self.assertEqual(len(grid), count)
            self.assertEqual(len({row["experiment_id"] for row in grid}), count)
            self.assertFalse(any(row["budget"] == 0 and row["reasoning_language"] != row["language"] for row in grid))

    def test_smoke_and_pilot_outputs_can_be_resumed_in_development(self):
        development = {row["experiment_id"]: row for row in build_grid(self.config, self.manifest, self.records, "development", self.current_hash)}
        for row in build_grid(self.config, self.manifest, self.records, "pilot", self.current_hash):
            self.assertEqual(row, development[row["experiment_id"]])

    def test_gold_never_enters_prompt_or_job_record(self):
        prompt = render_prompt("How much is 2 + 3?", "bn", 4, "en")
        self.assertIn("in English", prompt)
        self.assertIn("in Bengali", prompt)
        for job in build_grid(self.config, self.manifest, self.records, "smoke", self.current_hash):
            self.assertNotIn("gold", job)

    def test_fingerprint_changes_when_settings_change(self):
        changed = deepcopy(self.config)
        changed["generation"]["temperature"] = 0.8
        self.assertNotEqual(context_hash(changed, self.manifest), self.current_hash)

    def test_invalid_splits_and_modified_records_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = deepcopy(self.manifest)
            manifest["splits"]["test"][0] = manifest["splits"]["development"][0]
            write_json(folder / "manifest.json", manifest)
            write_jsonl(folder / "records.jsonl", self.records)
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_data(self.config, folder)
            write_json(folder / "manifest.json", self.manifest)
            changed = deepcopy(self.records)
            changed[0]["gold"] = "99999"
            write_jsonl(folder / "records.jsonl", changed)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_data(self.config, folder)

    def test_duplicates_stale_outputs_and_test_access_are_rejected(self):
        job = build_grid(self.config, self.manifest, self.records, "smoke", self.current_hash)[0]
        row = fixture_response(job)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_jsonl(folder / "raw/sample.jsonl", [row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_scored(self.config, self.manifest, self.records, folder)
            stale = deepcopy(row)
            stale["config_hash"] = "changed"
            write_jsonl(folder / "raw/sample.jsonl", [stale])
            with self.assertRaisesRegex(ValueError, "Mismatched"):
                load_scored(self.config, self.manifest, self.records, folder)
            test_job = build_grid(self.config, self.manifest, self.records, "test", self.current_hash)[0]
            write_jsonl(folder / "raw/sample.jsonl", [fixture_response(test_job)])
            with self.assertRaisesRegex(ValueError, "locked"):
                load_scored(self.config, self.manifest, self.records, folder)

    def test_incomplete_blocks_do_not_enter_balanced_curves(self):
        grid = build_grid(self.config, self.manifest, self.records, "smoke", self.current_hash)
        first_id = grid[0]["problem_id"]
        one_block = [fixture_response(job) for job in grid if job["problem_id"] == first_id]
        selected, ids = balanced_rows(self.config, self.manifest, self.records, one_block, "smoke")
        self.assertEqual(len(selected), 46)
        self.assertEqual(ids, [first_id])
        selected, ids = balanced_rows(self.config, self.manifest, self.records, one_block[:-1], "smoke")
        self.assertEqual((selected, ids), ([], []))
        report = coverage(self.config, self.manifest, self.records, one_block)
        self.assertEqual(report["overall"]["complete"], 46)
        self.assertEqual(report["overall"]["expected"], 11500)
        self.assertEqual(report["smoke"]["complete_paired_ids"], 1)

    def test_paired_contrasts_preserve_problem_pairs(self):
        rows = []
        for problem_id, short, long in [("a", True, False), ("b", False, True)]:
            for budget, correct in [(4, short), (16, long)]:
                rows.append({"model": "small", "problem_id": problem_id, "language": "en", "reasoning_language": "en", "budget": budget, "correct": correct})
        result = paired_contrasts(rows, self.config)
        self.assertEqual(result[0]["n"], 2)
        self.assertEqual(result[0]["accuracy_difference"], 0)
        self.assertEqual(bootstrap_interval([1.0, 1.0]), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
