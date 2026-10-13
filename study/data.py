"""Pinned original MGSM files, row alignment checks, and shared-ID splits."""
from __future__ import annotations

import csv
import io
import random
import urllib.request
from pathlib import Path

from .common import digest, file_digest, read_json, read_jsonl, write_json, write_jsonl
from .scoring import normalize_number


def prepare_data(config: dict, data_dir: Path) -> dict:
    raw_dir = data_dir / "raw" / config["dataset_revision"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[list[str]]] = {}
    sources = {}
    for language in config["languages"]:
        path = raw_dir / f"mgsm_{language}.tsv"
        url = f"https://raw.githubusercontent.com/google-research/url-nlp/{config['dataset_revision']}/mgsm/mgsm_{language}.tsv"
        if not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "indic-reasoning-study/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
            temporary = path.with_suffix(".download")
            temporary.write_bytes(content)
            temporary.replace(path)
        rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8")), delimiter="\t"))
        if len(rows) != 250 or any(len(row) != 2 or not row[0].strip() for row in rows):
            raise ValueError(f"Expected 250 question/answer pairs in {path}")
        all_rows[language] = rows
        sources[language] = {"url": url, "sha256": file_digest(path)}
    ids = [f"mgsm-{number:03d}" for number in range(250)]
    shuffled = ids.copy()
    random.Random(config["split_seed"]).shuffle(shuffled)
    development = shuffled[:config["development_size"]]
    test = shuffled[config["development_size"]:]
    records = []
    for number, problem_id in enumerate(ids):
        golds = {normalize_number(all_rows[lang][number][1]) for lang in config["languages"]}
        if len(golds) != 1:
            raise ValueError(f"Translations disagree on the answer at row {number + 1}; investigate alignment")
        for language in config["languages"]:
            records.append({"problem_id": problem_id, "source_row": number + 1,
                            "language": language, "question": all_rows[language][number][0],
                            "gold": next(iter(golds)),
                            "split": "development" if problem_id in development else "test"})
    manifest = {
        "schema_version": 1, "dataset_revision": config["dataset_revision"],
        "alignment": "Shared zero-based row IDs in the original parallel release; numeric answers checked across all rows. Equal answers alone do not prove semantic alignment.",
        "sources": sources, "dataset_hash": digest(records), "record_count": len(records),
        "split_seed": config["split_seed"],
        "splits": {"development": development, "pilot": development[:config["pilot_size"]],
                   "smoke": development[:config["smoke_size"]], "test": test}
    }
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Existing manifest differs; use a new data directory rather than overwriting frozen splits")
    write_jsonl(data_dir / "records.jsonl", records)
    write_json(manifest_path, manifest)
    return manifest


def load_data(config: dict, data_dir: Path) -> tuple[dict, list[dict]]:
    manifest = read_json(data_dir / "manifest.json")
    records = read_jsonl(data_dir / "records.jsonl")
    if manifest["dataset_revision"] != config["dataset_revision"] or digest(records) != manifest["dataset_hash"]:
        raise ValueError("Dataset revision or record checksum changed")
    if manifest["split_seed"] != config["split_seed"]:
        raise ValueError("Split seed differs from the data manifest")
    expected_sizes = {"development": config["development_size"], "pilot": config["pilot_size"],
                      "smoke": config["smoke_size"], "test": 250 - config["development_size"]}
    for phase, size in expected_sizes.items():
        if len(manifest["splits"][phase]) != size:
            raise ValueError(f"Unexpected {phase} split size")
    development, test = map(set, (manifest["splits"]["development"], manifest["splits"]["test"]))
    if development & test or len(development | test) != 250:
        raise ValueError("Problem IDs overlap or are missing")
    if not set(manifest["splits"]["pilot"]) <= development or not set(manifest["splits"]["smoke"]) <= set(manifest["splits"]["pilot"]):
        raise ValueError("Pilot/smoke IDs must be nested development subsets")
    for row in records:
        expected = "development" if row["problem_id"] in development else "test"
        if row["split"] != expected:
            raise ValueError("Record split differs from frozen ID split")
    return manifest, records
