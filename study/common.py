"""Small, dependency-free file and experiment utilities."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{number}; recover the partial last line before resuming") from error
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def load_config(path: Path | None = None) -> dict:
    config = read_json(path or ROOT / "configs/study.json")
    if config["languages"] != ["en", "bn", "te"] or config["budgets"] != [0, 2, 4, 8, 16]:
        raise ValueError("This study requires en/bn/te and the five preregistered budgets")
    if config["generation"]["enable_thinking"]:
        raise ValueError("Native thinking must remain disabled")
    if config["generation"]["quantization"] not in {"none", "4bit"}:
        raise ValueError("quantization must be none or 4bit")
    if config["generation"]["dtype"] != "float16":
        raise ValueError("The implemented inference backend uses float16")
    if set(config["models"]) != {"qwen3-1.7b", "qwen3-4b"}:
        raise ValueError("This study requires the two declared Qwen3 model aliases")
    if (config["development_size"], config["pilot_size"], config["smoke_size"]) != (100, 50, 10):
        raise ValueError("This study requires the declared 100/50/10 development subsets")
    return config


def source_fingerprint() -> str:
    # Editing a prompt, scorer, feature, or other pipeline code invalidates old experiments.
    return digest({path.name: file_digest(path) for path in sorted((ROOT / "study").glob("*.py"))})


def context_hash(config: dict, manifest: dict) -> str:
    return digest({"config": config, "dataset": manifest["dataset_hash"],
                   "splits": manifest["splits"], "code": source_fingerprint()})
