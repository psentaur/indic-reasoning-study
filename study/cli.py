"""Run with python -m study or indic-study. Preparation/scoring need only Python."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .common import ROOT, context_hash, load_config, write_json, write_jsonl
from .data import load_data, prepare_data
from .prompts import build_grid
from .results import coverage, load_scored


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Reproducible MGSM explanation-budget study")
    value.add_argument("--config", type=Path, default=ROOT / "configs/study.json")
    value.add_argument("--data-dir", type=Path, default=ROOT / "data")
    value.add_argument("--results-dir", type=Path, default=ROOT / "results")
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Download pinned original MGSM data and freeze shared-ID splits")
    commands.add_parser("characterize", help="Measure development input tokenization without GPU inference")
    planning = commands.add_parser("plan", help="Export prompts and workload; does not call a model")
    planning.add_argument("--phase", choices=["smoke", "pilot", "development", "test"], default="pilot")
    planning.add_argument("--model", choices=["qwen3-1.7b", "qwen3-4b"])
    planning.add_argument("--output", type=Path)
    running = commands.add_parser("run", help="Execute/resume real GPU inference")
    running.add_argument("--phase", choices=["smoke", "pilot", "development", "test"], required=True)
    running.add_argument("--model", choices=["qwen3-1.7b", "qwen3-4b"])
    running.add_argument("--limit", type=int, help="Maximum NEW responses; prior completed responses are skipped")
    commands.add_parser("status", help="Validate logs and report exact coverage")
    scoring = commands.add_parser("score", help="Validate and numerically score real inference outputs")
    scoring.add_argument("--split", choices=["development", "test"])
    analysis = commands.add_parser("analyze", help="Export balanced curves and paired intervals")
    analysis.add_argument("--phase", choices=["smoke", "pilot", "development", "test"], default="pilot")
    analysis.add_argument("--model", choices=["qwen3-1.7b", "qwen3-4b"])
    review = commands.add_parser("review-sample", help="Export a stratified incorrect-response review sheet")
    review.add_argument("--size", type=int, default=40)
    review.add_argument("--split", choices=["development", "test"], default="development")
    commands.add_parser("train-router", help="Grouped dev validation, full dev fit, and held-out freeze")
    commands.add_parser("evaluate-router", help="Evaluate frozen router on complete cached held-out outputs")
    route = commands.add_parser("route", help="Choose a budget using only the question")
    route.add_argument("--model", choices=["qwen3-1.7b", "qwen3-4b"], required=True)
    route.add_argument("--language", choices=["en", "bn", "te"], required=True)
    route.add_argument("--question", required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "prepare":
            manifest = prepare_data(config, args.data_dir)
            output = {"records": manifest["record_count"], "dataset_hash": manifest["dataset_hash"],
                      "split_sizes": {phase: len(ids) for phase, ids in manifest["splits"].items()}}
        else:
            manifest, records = load_data(config, args.data_dir)
            if args.command == "characterize":
                from .characterization import characterize
                output = characterize(config, manifest, records, args.data_dir / "characterization")
            elif args.command == "plan":
                grid = build_grid(config, manifest, records, args.phase, context_hash(config, manifest), args.model)
                if args.output:
                    write_jsonl(args.output, grid)
                output = {"phase": args.phase, "responses": len(grid),
                          "problem_ids": len({row["problem_id"] for row in grid}),
                          "config_hash": context_hash(config, manifest),
                          "sample_prompt": grid[0]["prompt"], "model_calls": 0}
            elif args.command == "run":
                from .inference import run
                output = run(config, manifest, records, args.results_dir, args.phase, args.model, args.limit)
            elif args.command == "route":
                from .router import route_question
                output = route_question(config, manifest, args.results_dir, args.model, args.language, args.question)
            else:
                rows = load_scored(config, manifest, records, args.results_dir)
                if args.command == "status":
                    output = coverage(config, manifest, records, rows)
                elif args.command == "score":
                    selected = [row for row in rows if args.split is None or row["split"] == args.split]
                    output_path = args.results_dir / "scored" / f"{args.split or 'all'}.jsonl"
                    write_jsonl(output_path, selected)
                    output = {"responses": len(selected), "correct": sum(row["correct"] for row in selected),
                              "parse_failures": sum(row["parse_error"] is not None for row in selected),
                              "output": str(output_path), "coverage": coverage(config, manifest, records, rows)}
                elif args.command == "analyze":
                    from .analysis import analyze
                    output_path = args.results_dir / "analysis" / args.phase / (args.model or "both-models")
                    output = analyze(config, manifest, records, rows, output_path, args.phase, args.model)
                elif args.command == "review-sample":
                    from .analysis import review_sample
                    if args.size < 1:
                        raise ValueError("Review sample size must be positive")
                    output_path = args.results_dir / "analysis" / f"{args.split}_review.csv"
                    selected = [row for row in rows if row["split"] == args.split]
                    if selected and all(row["correct"] for row in selected):
                        output = {"sampled": 0, "note": "No incorrect completed responses to review"}
                    else:
                        output = {"sampled": review_sample(records, selected, output_path, args.size), "output": str(output_path)}
                elif args.command == "train-router":
                    from .router import train
                    output = train(config, manifest, records, rows, args.results_dir)
                elif args.command == "evaluate-router":
                    from .router import evaluate_test
                    output = evaluate_test(config, manifest, records, rows, args.results_dir)
                else:
                    raise ValueError("Unknown command")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
