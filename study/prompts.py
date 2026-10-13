"""Matched English instructions; gold answers never enter a prompt."""
from __future__ import annotations

from .common import digest

LANGUAGE_NAMES = {"en": "English", "bn": "Bengali", "te": "Telugu"}
PROMPT_VERSION = "1"


def conditions(language: str, budgets: list[int]) -> list[tuple[int, str]]:
    output = []
    for budget in budgets:
        output.append((budget, language))
        if budget > 0 and language != "en":
            output.append((budget, "en"))
    return output


def render_prompt(question: str, language: str, budget: int, reasoning_language: str) -> str:
    final_language = LANGUAGE_NAMES[language]
    if budget == 0:
        instructions = "Give a direct answer. Do not show an explanation or numbered steps."
    else:
        instructions = (f"Explain the solution in {LANGUAGE_NAMES[reasoning_language]} using exactly {budget} numbered steps, "
                        "with one step per line in the format '1. ...'. Do not add other explanation paragraphs.")
    return ("Solve the following mathematical word problem.\n"
            f"{instructions}\n"
            f"Give a brief final answer sentence in {final_language}.\n"
            "Then put only the numerical answer on the last line in the exact format FINAL_ANSWER: <number>. "
            "Do not include units or any other text on that last line.\n\n"
            f"Problem:\n{question}")


def build_grid(config: dict, manifest: dict, records: list[dict], phase: str,
               config_hash: str, model_filter: str | None = None) -> list[dict]:
    ids = manifest["splits"][phase]
    lookup = {(row["problem_id"], row["language"]): row for row in records}
    output = []
    for model in config["models"]:
        if model_filter and model != model_filter:
            continue
        for problem_id in ids:
            for language in config["languages"]:
                question = lookup[problem_id, language]["question"]
                for budget, reasoning_language in conditions(language, config["budgets"]):
                    key = {"problem_id": problem_id, "language": language, "model": model,
                           "budget": budget, "reasoning_language": reasoning_language,
                           "config_hash": config_hash}
                    experiment_id = digest(key)
                    output.append({**key, "experiment_id": experiment_id,
                                   "model_name": config["models"][model]["name"],
                                   "model_revision": config["models"][model]["revision"],
                                   "split": "test" if phase == "test" else "development",
                                   "prompt_version": PROMPT_VERSION,
                                   "prompt": render_prompt(question, language, budget, reasoning_language),
                                   "seed": (config["generation"]["seed"] + int(experiment_id[:8], 16)) % (2**32)})
    return output
