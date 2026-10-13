"""Validate logged outputs against the current experiment before using them."""
from __future__ import annotations

from pathlib import Path

from .common import context_hash, digest, file_digest, read_json, read_jsonl
from .prompts import build_grid
from .scoring import score_response


def verify_freeze(path: Path, expected_hash: str) -> dict:
    if not path.exists():
        raise ValueError("Held-out evaluation is locked. Complete development runs and train the router first.")
    freeze = read_json(path)
    if freeze["config_hash"] != expected_hash:
        raise ValueError("Config, dataset, splits, or pipeline code changed after freeze")
    for artifact in freeze["artifacts"]:
        artifact_path = path.parent / artifact["path"]
        if not artifact_path.exists() or file_digest(artifact_path) != artifact["sha256"]:
            raise ValueError(f"Frozen router artifact changed: {artifact_path}")
    return freeze


def load_scored(config: dict, manifest: dict, records: list[dict], results_dir: Path,
                split: str | None = None) -> list[dict]:
    current_hash = context_hash(config, manifest)
    freeze_path = results_dir / "freeze.json"
    freeze = verify_freeze(freeze_path, current_hash) if freeze_path.exists() else None
    expected = {row["experiment_id"]: row for phase in ["development", "test"]
                for row in build_grid(config, manifest, records, phase, current_hash)}
    golds = {(row["problem_id"], row["language"]): row["gold"] for row in records}
    output = []
    seen = set()
    for path in sorted((results_dir / "raw").glob("*.jsonl")):
        if path.name.endswith(".errors.jsonl"):
            continue
        for row in read_jsonl(path):
            if row.get("status") != "complete" or row.get("backend") != "transformers":
                raise ValueError(f"Unrecognized/non-completed inference record in {path}")
            key = row.get("experiment_id")
            if key not in expected:
                raise ValueError(f"Unknown or stale experiment in {path}: {key}")
            planned = expected[key]
            for field in planned:
                if row.get(field) != planned[field]:
                    raise ValueError(f"Mismatched {field} in experiment {key}")
            if key in seen:
                raise ValueError(f"Duplicate experiment {key}; duplicates must not inflate coverage")
            seen.add(key)
            if not isinstance(row.get("response"), str) or not isinstance(row.get("output_tokens"), int) or row["output_tokens"] < 0:
                raise ValueError(f"Invalid output fields in {key}")
            if row["split"] == "test":
                if freeze is None:
                    freeze = verify_freeze(freeze_path, current_hash)
            output.append(score_response(row, golds[row["problem_id"], row["language"]]))
    if freeze is not None:
        development = sorted([row for row in output if row["split"] == "development"], key=lambda row: row["experiment_id"])
        if digest(development) != freeze["development_results_hash"]:
            raise ValueError("Development outputs changed after the study was frozen")
    return [row for row in output if split is None or row["split"] == split]


def coverage(config: dict, manifest: dict, records: list[dict], rows: list[dict]) -> dict:
    current_hash = context_hash(config, manifest)
    seen = {row["experiment_id"] for row in rows}
    output = {}
    for phase in ["smoke", "pilot", "development", "test"]:
        grid = build_grid(config, manifest, records, phase, current_hash)
        keys = {row["experiment_id"] for row in grid}
        per_id = {}
        for row in grid:
            per_id.setdefault(row["problem_id"], set()).add(row["experiment_id"])
        count = len(keys & seen)
        output[phase] = {"expected": len(keys), "complete": count,
                         "fraction": count / len(keys),
                         "complete_paired_ids": sum(keys_for_id <= seen for keys_for_id in per_id.values())}
    total = output["development"]["expected"] + output["test"]["expected"]
    output["overall"] = {"expected": total, "complete": len(seen), "fraction": len(seen) / total}
    return output
