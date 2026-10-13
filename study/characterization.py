"""Real development-data tokenization measurements; never model accuracy results."""
from __future__ import annotations

from collections import defaultdict
import importlib.metadata
import statistics
from pathlib import Path

from .analysis import bootstrap_interval, write_csv
from .common import write_json
from .prompts import render_prompt


def characterize(config: dict, manifest: dict, records: list[dict], output_dir: Path) -> dict:
    from transformers import AutoTokenizer

    development_ids = set(manifest["splits"]["development"])
    development = [row for row in records if row["problem_id"] in development_ids]
    measurements = []
    for model, specification in config["models"].items():
        tokenizer = AutoTokenizer.from_pretrained(specification["name"], revision=specification["revision"])
        for row in development:
            prompt = render_prompt(row["question"], row["language"], 16, row["language"])
            chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
            measurements.append({"model": model, "tokenizer_revision": specification["revision"],
                                 "problem_id": row["problem_id"], "language": row["language"],
                                 "question_characters": len(row["question"]),
                                 "question_tokens": len(tokenizer.encode(row["question"], add_special_tokens=False)),
                                 "native_16_step_prompt_tokens": len(tokenizer.encode(chat, add_special_tokens=False))})
    summary = []
    english = {(row["model"], row["problem_id"]): row["question_tokens"] for row in measurements if row["language"] == "en"}
    groups = defaultdict(list)
    for row in measurements:
        groups[row["model"], row["language"]].append(row)
    for (model, language), group in sorted(groups.items()):
        ratios = [row["question_tokens"] / english[model, row["problem_id"]] for row in sorted(group, key=lambda row: row["problem_id"])]
        low, high = bootstrap_interval(ratios, **config["bootstrap"])
        summary.append({"model": model, "language": language, "development_ids": len(group),
                        "mean_question_tokens": statistics.mean(row["question_tokens"] for row in group),
                        "median_question_tokens": statistics.median(row["question_tokens"] for row in group),
                        "mean_paired_token_ratio_to_english": statistics.mean(ratios),
                        "ratio_ci_low": low, "ratio_ci_high": high})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "measurements.csv", measurements)
    write_csv(output_dir / "summary.csv", summary)
    report = {"kind": "development_input_tokenization_only", "dataset_hash": manifest["dataset_hash"],
              "problem_ids": len(development_ids), "gold_answers_used": False, "model_generations": 0,
              "environment": {name: importlib.metadata.version(name) for name in ["transformers", "tokenizers", "jinja2"]},
              "summary": summary,
              "interpretation": "These are actual input token counts. They do not measure output length, mathematical accuracy, reasoning quality, or a capacity cliff, and do not count toward the 2,300-response milestone."}
    write_json(output_dir / "report.json", report)
    lines = ["# Development input tokenization", "", report["interpretation"], "",
             "| Model | Language | IDs | Mean question tokens | Mean paired ratio to English |",
             "|---|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['model']} | {row['language']} | {row['development_ids']} | {row['mean_question_tokens']:.2f} | {row['mean_paired_token_ratio_to_english']:.2f} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    values = {row["language"]: row for row in summary if row["model"] == next(iter(config["models"]))}
    languages = ["en", "bn", "te"]
    ratios = [values[language]["mean_paired_token_ratio_to_english"] for language in languages]
    figure, axis = plt.subplots(figsize=(3.5, 2.8))
    axis.bar(["English", "Bengali", "Telugu"], ratios, color=["#536878", "#148a92", "#b86b37"],
             edgecolor="black", linewidth=0.6,
             yerr=[[ratios[index] - values[language]["ratio_ci_low"] for index, language in enumerate(languages)],
                   [values[language]["ratio_ci_high"] - ratios[index] for index, language in enumerate(languages)]], capsize=4)
    for index, ratio in enumerate(ratios):
        axis.text(index, ratio + 0.28, f"{ratio:.2f}×", ha="center", fontsize=11)
    axis.set_ylabel("Paired input-token ratio to English", fontsize=10)
    axis.set_title("Development questions (n=100)\nQwen3 tokenizers", fontsize=10)
    axis.set_ylim(0, max(ratios) + 1.1)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output_dir / "input_token_ratios.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "input_token_ratios.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return report
