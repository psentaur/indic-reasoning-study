"""Question-only routing with grouped development validation and frozen test evaluation."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import re
from pathlib import Path

from .analysis import bootstrap_interval, write_csv
from .common import context_hash, digest, file_digest, read_json, write_json
from .results import verify_freeze


def features(question: str, language: str, include_language: bool = True) -> dict:
    value = {"character_length": len(question),
             "sentence_count": max(1, len(re.findall(r"[.!?।]+(?:\s|$)", question))),
             "quantity_count": len(re.findall(r"\d+(?:[.,]\d+)*", question))}
    if include_language:
        value["language"] = language
    return value


def native_lookup(rows: list[dict], model: str) -> dict:
    return {(row["problem_id"], row["language"], row["budget"]): row
            for row in rows if row["model"] == model and row["reasoning_language"] == row["language"]}


def best_fixed(lookup: dict, questions: list[dict], budgets: list[int]) -> int:
    return max(budgets, key=lambda budget: (sum(lookup[row["problem_id"], row["language"], budget]["correct"]
                                               for row in questions), -budget))


def fit_policy(lookup: dict, questions: list[dict], config: dict, include_language: bool = True) -> dict:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeClassifier

    global_budget = best_fixed(lookup, questions, config["budgets"])
    language_budgets = {language: best_fixed(lookup, [row for row in questions if row["language"] == language], config["budgets"])
                        for language in config["languages"]}
    training_features, labels = [], []
    for row in questions:
        successful = [budget for budget in config["budgets"] if lookup[row["problem_id"], row["language"], budget]["correct"]]
        if successful:
            training_features.append(features(row["question"], row["language"], include_language))
            labels.append(min(successful))
    estimator = None
    if labels:
        estimator = make_pipeline(DictVectorizer(sparse=False),
                                  DecisionTreeClassifier(max_depth=config["router"]["max_depth"],
                                                         min_samples_leaf=config["router"]["min_samples_leaf"],
                                                         random_state=config["router"]["seed"]))
        estimator.fit(training_features, labels)
    return {"estimator": estimator, "include_language": include_language,
            "global_budget": global_budget, "language_budgets": language_budgets,
            "training_questions": len(questions), "labeled_questions": len(labels),
            "excluded_no_success": len(questions) - len(labels),
            "policy_kind": "decision_tree" if labels else "language_lookup_fallback_no_successful_labels"}


def predict(policy: dict, question: str, language: str) -> int:
    if language not in policy["language_budgets"]:
        raise ValueError("Only en, bn, and te are supported")
    if policy["estimator"] is None:
        return policy["language_budgets"][language]
    return int(policy["estimator"].predict([features(question, language, policy["include_language"])])[0])


def evaluate_policies(lookup: dict, questions: list[dict], policy: dict, ablation: dict,
                      model: str, fold: int | None = None) -> list[dict]:
    output = []
    for question in questions:
        choices = {"router": predict(policy, question["question"], question["language"]),
                   "router_without_language": predict(ablation, question["question"], question["language"]),
                   "always_8": 8, "dev_best_global": policy["global_budget"],
                   "dev_best_language": policy["language_budgets"][question["language"]]}
        for name, budget in choices.items():
            row = lookup[question["problem_id"], question["language"], budget]
            output.append({"model": model, "problem_id": question["problem_id"], "language": question["language"],
                           "policy": name, "budget": budget, "correct": int(row["correct"]),
                           "output_tokens": row["output_tokens"], "fold": fold})
    return output


def aggregate_metrics(evaluations: list[dict], config: dict) -> dict:
    groups = defaultdict(list)
    for row in evaluations:
        groups[row["model"], row["policy"]].append(row)
    summary = []
    for (model, policy), rows in sorted(groups.items()):
        by_id = defaultdict(list)
        for row in rows:
            by_id[row["problem_id"]].append(row["correct"])
        low, high = bootstrap_interval([sum(values) / len(values) for _, values in sorted(by_id.items())], **config["bootstrap"])
        summary.append({"model": model, "policy": policy, "questions": len(rows), "problem_ids": len(by_id),
                        "accuracy": sum(row["correct"] for row in rows) / len(rows),
                        "accuracy_ci_low": low, "accuracy_ci_high": high,
                        "mean_output_tokens": sum(row["output_tokens"] for row in rows) / len(rows)})
    lookup = {(row["model"], row["problem_id"], row["language"], row["policy"]): row for row in evaluations}
    contrasts = []
    for model in sorted({row["model"] for row in evaluations}):
        questions = sorted({(row["problem_id"], row["language"]) for row in evaluations if row["model"] == model})
        for baseline in ["always_8", "dev_best_global", "dev_best_language", "router_without_language"]:
            paired_by_id = defaultdict(list)
            token_by_id = defaultdict(list)
            for problem_id, language in questions:
                router = lookup[model, problem_id, language, "router"]
                other = lookup[model, problem_id, language, baseline]
                paired_by_id[problem_id].append(router["correct"] - other["correct"])
                token_by_id[problem_id].append(router["output_tokens"] - other["output_tokens"])
            accuracy_deltas = [sum(values) / len(values) for _, values in sorted(paired_by_id.items())]
            token_deltas = [sum(values) / len(values) for _, values in sorted(token_by_id.items())]
            low, high = bootstrap_interval(accuracy_deltas, **config["bootstrap"])
            token_low, token_high = bootstrap_interval(token_deltas, **config["bootstrap"])
            contrasts.append({"model": model, "baseline": baseline, "problem_ids": len(paired_by_id),
                              "accuracy_difference": sum(accuracy_deltas) / len(accuracy_deltas),
                              "ci_low": low, "ci_high": high,
                              "mean_token_difference": sum(token_deltas) / len(token_deltas),
                              "token_difference_ci_low": token_low, "token_difference_ci_high": token_high})
    language_summary = []
    language_groups = defaultdict(list)
    for row in evaluations:
        language_groups[row["model"], row["policy"], row["language"]].append(row)
    for (model, policy, language), rows in sorted(language_groups.items()):
        low, high = bootstrap_interval([row["correct"] for row in sorted(rows, key=lambda row: row["problem_id"])], **config["bootstrap"])
        language_summary.append({"model": model, "policy": policy, "language": language, "questions": len(rows),
                                 "accuracy": sum(row["correct"] for row in rows) / len(rows),
                                 "accuracy_ci_low": low, "accuracy_ci_high": high,
                                 "mean_output_tokens": sum(row["output_tokens"] for row in rows) / len(rows)})
    return {"summary": summary, "language_summary": language_summary, "paired_contrasts": contrasts}


def train(config: dict, manifest: dict, records: list[dict], rows: list[dict], results_dir: Path) -> dict:
    import joblib
    import numpy as np
    from sklearn.model_selection import GroupKFold
    import sklearn

    if (results_dir / "freeze.json").exists():
        raise ValueError("This study is already frozen. Do not retrain after test access; start a new study directory for a revised design.")
    if any(row["split"] == "test" for row in rows):
        raise ValueError("Router training cannot consume held-out results")
    development = set(manifest["splits"]["development"])
    questions = [row for row in records if row["problem_id"] in development]
    # Require the entire declared development grid, including pivot runs, before locking test access.
    if len(rows) != len(development) * len(config["models"]) * 23:
        raise ValueError("Complete all 4,600 development responses before freezing the study")
    output_dir = results_dir / "router"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts, model_details, validation = [], {}, []
    groups = np.asarray([row["problem_id"] for row in questions])
    folds = list(GroupKFold(n_splits=config["router"]["folds"]).split(questions, groups=groups))
    for model in config["models"]:
        lookup = native_lookup(rows, model)
        required = {(row["problem_id"], row["language"], budget) for row in questions for budget in config["budgets"]}
        if not required <= lookup.keys():
            raise ValueError(f"Incomplete native development grid for {model}")
        for fold, (training_indices, validation_indices) in enumerate(folds):
            training_questions = [questions[index] for index in training_indices]
            validation_questions = [questions[index] for index in validation_indices]
            policy = fit_policy(lookup, training_questions, config)
            ablation = fit_policy(lookup, training_questions, config, include_language=False)
            validation += evaluate_policies(lookup, validation_questions, policy, ablation, model, fold)
        policy = fit_policy(lookup, questions, config)
        ablation = fit_policy(lookup, questions, config, include_language=False)
        artifact_path = output_dir / f"{model}.joblib"
        joblib.dump({"config_hash": context_hash(config, manifest), "model": model,
                     "policy": policy, "ablation": ablation, "sklearn_version": sklearn.__version__}, artifact_path)
        artifacts.append({"path": str(artifact_path.relative_to(results_dir)), "sha256": file_digest(artifact_path)})
        model_details[model] = {key: value for key, value in policy.items() if key != "estimator"}
    metrics = {"phase": "grouped_development_cross_validation", "models": model_details,
               **aggregate_metrics(validation, config),
               "label_note": "Shortest successful budget from one sampled response per condition; labels may be noisy.",
               "selection_note": "All baseline fitting and router fitting occur within each training fold. No test results used."}
    write_json(output_dir / "development_metrics.json", metrics)
    write_csv(output_dir / "development_summary.csv", metrics["summary"])
    write_csv(output_dir / "development_language_summary.csv", metrics["language_summary"])
    write_csv(output_dir / "development_predictions.csv", validation)
    artifacts.append({"path": "router/development_metrics.json", "sha256": file_digest(output_dir / "development_metrics.json")})
    frozen = {"schema_version": 1, "config_hash": context_hash(config, manifest),
              "created_at": datetime.now(timezone.utc).isoformat(), "artifacts": artifacts,
              "development_results_hash": digest(sorted(rows, key=lambda row: row["experiment_id"])),
              "development_responses": len(rows), "test_ids": manifest["splits"]["test"],
              "router_scope": "Native explanation budgets only; pivot routing is out of scope"}
    write_json(results_dir / "freeze.json", frozen)
    return metrics


def evaluate_test(config: dict, manifest: dict, records: list[dict], rows: list[dict], results_dir: Path) -> dict:
    import joblib
    import sklearn

    freeze = verify_freeze(results_dir / "freeze.json", context_hash(config, manifest))
    development_rows = [row for row in rows if row["split"] == "development"]
    if digest(sorted(development_rows, key=lambda row: row["experiment_id"])) != freeze["development_results_hash"]:
        raise ValueError("Development outputs changed after freeze")
    test_rows = [row for row in rows if row["split"] == "test"]
    if len(test_rows) != len(manifest["splits"]["test"]) * len(config["models"]) * 23:
        raise ValueError("Complete all 6,900 held-out responses before final router evaluation")
    questions = [row for row in records if row["split"] == "test"]
    evaluations = []
    for model in config["models"]:
        artifact = joblib.load(results_dir / f"router/{model}.joblib")
        if artifact["sklearn_version"] != sklearn.__version__:
            raise ValueError("Restore the sklearn version used to fit the frozen router")
        evaluations += evaluate_policies(native_lookup(test_rows, model), questions,
                                         artifact["policy"], artifact["ablation"], model)
    metrics = {"phase": "held_out_test", **aggregate_metrics(evaluations, config),
               "notes": "Frozen question-only policies select cached native outputs; no additional LLM calls."}
    write_json(results_dir / "router/test_metrics.json", metrics)
    write_csv(results_dir / "router/test_summary.csv", metrics["summary"])
    write_csv(results_dir / "router/test_language_summary.csv", metrics["language_summary"])
    write_csv(results_dir / "router/test_paired_contrasts.csv", metrics["paired_contrasts"])
    write_csv(results_dir / "router/test_predictions.csv", evaluations)
    return metrics


def route_question(config: dict, manifest: dict, results_dir: Path, model: str,
                   language: str, question: str) -> dict:
    import joblib
    import sklearn
    verify_freeze(results_dir / "freeze.json", context_hash(config, manifest))
    artifact = joblib.load(results_dir / f"router/{model}.joblib")
    if artifact["sklearn_version"] != sklearn.__version__:
        raise ValueError("Restore the sklearn version used to fit the frozen router")
    return {"model": model, "language": language, "budget": predict(artifact["policy"], question, language),
            "reasoning_language": language, "features": features(question, language)}
