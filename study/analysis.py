"""Balanced-cohort accuracy curves, paired contrasts, and review samples."""
from __future__ import annotations

from collections import defaultdict
import csv
import random
from pathlib import Path

from .common import context_hash, write_json
from .prompts import build_grid
from .results import coverage


def bootstrap_interval(values: list[float], samples: int = 2000, seed: int = 42) -> tuple[float, float]:
    import numpy as np
    if not values:
        raise ValueError("Cannot bootstrap an empty cohort")
    values_array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = values_array[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def balanced_rows(config: dict, manifest: dict, records: list[dict], rows: list[dict],
                  phase: str, model_filter: str | None = None) -> tuple[list[dict], list[str]]:
    grid = build_grid(config, manifest, records, phase, context_hash(config, manifest), model_filter)
    expected = defaultdict(set)
    for row in grid:
        expected[row["problem_id"]].add(row["experiment_id"])
    available = {row["experiment_id"] for row in rows}
    complete_ids = sorted(problem_id for problem_id, keys in expected.items() if keys <= available)
    chosen_keys = set().union(*(expected[problem_id] for problem_id in complete_ids)) if complete_ids else set()
    return [row for row in rows if row["experiment_id"] in chosen_keys], complete_ids


def summarize(rows: list[dict], config: dict) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["language"], row["reasoning_language"], row["budget"]].append(row)
    summary = []
    for (model, language, reasoning_language, budget), group in sorted(groups.items()):
        low, high = bootstrap_interval([int(row["correct"]) for row in group], **config["bootstrap"])
        summary.append({"model": model, "language": language, "reasoning_language": reasoning_language,
                        "budget": budget, "n": len(group),
                        "accuracy": sum(row["correct"] for row in group) / len(group),
                        "accuracy_ci_low": low, "accuracy_ci_high": high,
                        "mean_output_tokens": sum(row["output_tokens"] for row in group) / len(group),
                        "mean_actual_steps": sum(row["actual_steps"] for row in group) / len(group),
                        "step_compliance": sum(row["step_compliant"] for row in group) / len(group),
                        "parse_failure_rate": sum(row["parse_error"] is not None for row in group) / len(group),
                        "truncation_rate": sum(row["truncated"] for row in group) / len(group)})
    return summary


def paired_contrasts(rows: list[dict], config: dict) -> list[dict]:
    lookup = {(row["model"], row["problem_id"], row["language"], row["reasoning_language"], row["budget"]): row for row in rows}
    ids = sorted({row["problem_id"] for row in rows})
    contrasts = []

    def add(model: str, name: str, values: list[float]):
        if values:
            low, high = bootstrap_interval(values, **config["bootstrap"])
            contrasts.append({"model": model, "contrast": name, "n": len(values),
                              "accuracy_difference": sum(values) / len(values),
                              "ci_low": low, "ci_high": high,
                              "interpretation": "Exploratory paired 95% interval; no multiplicity-adjusted significance claim"})

    for model in sorted({row["model"] for row in rows}):
        native_deltas = {}
        for language in config["languages"]:
            deltas = {}
            for problem_id in ids:
                short = lookup.get((model, problem_id, language, language, 4))
                long = lookup.get((model, problem_id, language, language, 16))
                if short is not None and long is not None:
                    deltas[problem_id] = int(long["correct"]) - int(short["correct"])
            native_deltas[language] = deltas
            add(model, f"{language}: native 16 minus 4 steps", list(deltas.values()))
            if language != "en":
                for budget in config["budgets"][1:]:
                    values = []
                    for problem_id in ids:
                        native = lookup.get((model, problem_id, language, language, budget))
                        pivot = lookup.get((model, problem_id, language, "en", budget))
                        if native is not None and pivot is not None:
                            values.append(int(pivot["correct"]) - int(native["correct"]))
                    add(model, f"{language}: pivot minus native at {budget} steps", values)
        for language in ["bn", "te"]:
            shared = sorted(set(native_deltas[language]) & set(native_deltas["en"]))
            add(model, f"{language} minus en: difference in native 16-minus-4 effects",
                [native_deltas[language][problem_id] - native_deltas["en"][problem_id] for problem_id in shared])
    return contrasts


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to export to {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(summary: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    models = sorted({row["model"] for row in summary})
    figure, axes = plt.subplots(len(models), 3, figsize=(10, 3.3 * len(models)), squeeze=False, sharey=True)
    for model_index, model in enumerate(models):
        for language_index, language in enumerate(["en", "bn", "te"]):
            axis = axes[model_index, language_index]
            for reasoning_language in [language] + (["en"] if language != "en" else []):
                series = sorted([row for row in summary if row["model"] == model and row["language"] == language
                                 and row["reasoning_language"] == reasoning_language], key=lambda row: row["budget"])
                if not series:
                    continue
                pivot = reasoning_language != language
                axis.errorbar([row["budget"] for row in series], [100 * row["accuracy"] for row in series],
                              yerr=[[100 * (row["accuracy"] - row["accuracy_ci_low"]) for row in series],
                                    [100 * (row["accuracy_ci_high"] - row["accuracy"]) for row in series]],
                              linestyle="--" if pivot else "-", marker="s" if pivot else "o", capsize=3,
                              label="English pivot" if pivot else "Native")
            axis.set_title(f"{model} / {language.upper()}")
            axis.set_xticks([0, 2, 4, 8, 16])
            axis.set_xlabel("Requested explanation steps")
            axis.set_ylim(-2, 102)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        axes[model_index, 0].set_ylabel("Numerical accuracy (%)")
    figure.tight_layout()
    figure.savefig(output_dir / "accuracy_curves.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "accuracy_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze(config: dict, manifest: dict, records: list[dict], rows: list[dict],
            output_dir: Path, phase: str, model_filter: str | None = None) -> dict:
    selected, ids = balanced_rows(config, manifest, records, rows, phase, model_filter)
    if not selected:
        raise ValueError("No complete paired problem blocks yet. Complete every selected condition for at least one ID.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(selected, config)
    contrasts = paired_contrasts(selected, config)
    report = {"phase": phase, "model_filter": model_filter, "config_hash": context_hash(config, manifest),
              "cohort_ids": ids, "balanced_responses": len(selected),
              "coverage": coverage(config, manifest, records, rows), "summary": summary, "contrasts": contrasts,
              "notes": ["Only complete matched problem blocks enter these curves.",
                        "Requested steps are visible explanation instructions, not hidden compute measurements.",
                        "Native script counts do not establish language fluency or semantic correctness.",
                        "A negative contrast alone does not establish a capacity cliff."]}
    write_json(output_dir / "analysis.json", report)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "paired_contrasts.csv", contrasts)
    plot_curves(summary, output_dir)
    lines = [f"# {phase.title()} results", "", f"Balanced cohort: {len(ids)} problem IDs; {len(selected)} responses.", "",
             "| Model | Question | Explanation | Steps | N | Accuracy | Mean tokens | Truncated |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['model']} | {row['language']} | {row['reasoning_language']} | {row['budget']} | {row['n']} | {row['accuracy']:.1%} | {row['mean_output_tokens']:.1f} | {row['truncation_rate']:.1%} |")
    lines += ["", "## Interpretation checks", "", *[f"- {note}" for note in report["notes"]], "",
              "Write the observed insight after reviewing these results; no hypothesis is automatically marked confirmed."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"cohort_ids": len(ids), "responses": len(selected), "output_dir": str(output_dir)}


def review_sample(records: list[dict], rows: list[dict], output: Path, size: int = 40) -> int:
    questions = {row["problem_id"]: row["question"] for row in records if row["language"] == "en"}
    groups = defaultdict(list)
    for row in rows:
        if not row["correct"]:
            groups[row["model"], row["language"], row["reasoning_language"]].append(row)
    rng = random.Random(42)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < size and any(groups.values()):
        for group in groups.values():
            if group and len(selected) < size:
                selected.append(group.pop())
    if not selected:
        raise ValueError("No incorrect completed responses available for error review")
    output_rows = [{"experiment_id": row["experiment_id"], "problem_id": row["problem_id"],
                    "model": row["model"], "question_language": row["language"],
                    "explanation_language": row["reasoning_language"], "budget": row["budget"],
                    "english_question": questions[row["problem_id"]], "response": row["response"],
                    "parse_error": row["parse_error"], "truncated": row["truncated"],
                    "semantic_review_available": row["reasoning_language"] == "en",
                    "category": "", "reviewer": "", "notes": ""} for row in selected]
    write_csv(output, output_rows)
    return len(selected)
