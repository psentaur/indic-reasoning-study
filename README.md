# Indic reasoning study

Reproducible MGSM experiments on requested explanation length in English, Bengali, and Telugu, using pinned Qwen3-1.7B and Qwen3-4B checkpoints. The pipeline supports resumable inference, numerical scoring, paired analysis, and a question-only explanation-budget router.

This repository contains source code, tests, configurations, an optional Colab notebook, and supporting documentation. Dataset downloads, model weights, and experiment outputs are generated locally and ignored by Git.

## Install and prepare

Python 3.11 or newer is required; Python 3.12 is recommended. Preparation, workload planning, status, and scoring use the standard library. The `analysis` extra adds plotting and routing, `tokenizer` adds input-token characterization, and `gpu` adds inference libraries. Model inference requires a working CUDA GPU.

```bash
uv sync --extra analysis --extra tokenizer --python 3.12
source .venv/bin/activate

# Downloads the pinned original MGSM files and creates deterministic splits.
python -m study prepare
python -m study plan --phase pilot
python -m study status
python -m unittest discover -s tests -v
```

Run `prepare` before the tests: pipeline and router tests read the generated dataset. Initial preparation requires internet access. Alternatively, install with `python -m pip install -e '.[analysis,tokenizer]'` in a virtual environment. `uv.lock` records the resolved dependencies.

For CUDA inference, install `.[analysis,gpu]` with a compatible PyTorch build. The existing local runner requires a specific Linux CUDA environment; follow [the local setup guide](docs/LOCAL_RUN.md).

## Experiment workflow

All global options precede the subcommand. Defaults are `--config configs/study.json`, `--data-dir data`, and `--results-dir results`. Use a persistent results directory for long GPU runs:

```bash
python -m study --config configs/study.json --data-dir data --results-dir /path/to/results run --phase smoke
```

The phases share experiment IDs, so completed smoke and pilot responses are reused by development. The balanced workload is 460 smoke, 2,300 pilot, 4,600 development, and 6,900 held-out responses. Each problem has 23 conditions per model. See [the protocol](docs/PROTOCOL.md) for prompts, scoring, splits, and interpretation limits.

```bash
# CPU preparation and planning; no model calls.
python -m study prepare
python -m study plan --phase pilot --output /tmp/pilot-grid.jsonl
python -m study characterize  # Requires analysis and tokenizer extras.

# GPU feasibility and preliminary evaluation.
python -m study run --phase smoke --model qwen3-1.7b --limit 23
python -m study run --phase smoke --model qwen3-4b --limit 23
python -m study run --phase smoke
python -m study run --phase pilot
python -m study score --split development
python -m study analyze --phase pilot
python -m study review-sample --split development --size 40
python -m study status

# Full development, router freeze, and held-out evaluation.
python -m study run --phase development
python -m study analyze --phase development
python -m study train-router
python -m study run --phase test
python -m study score --split test
python -m study analyze --phase test
python -m study evaluate-router
python -m study review-sample --split test
python -m study route --model qwen3-1.7b --language en --question 'There are 12 eggs in each of 3 boxes. How many eggs?'
```

`--limit` limits new responses, rather than total responses. Use `--model` with `run`, `plan`, or `analyze` to select one checkpoint. If the larger model is unavailable, an alternative preliminary run is all 100 development IDs on `qwen3-1.7b` (2,300 responses); the two-model comparison and full router freeze remain incomplete.

`configs/study.json` uses FP16 weights. `configs/local-nf4.json` applies NF4 with double quantization and FP16 compute to both models. Choose precision before inference and keep each configuration in its own results directory.

## Outputs and reproducibility

Preparation writes `data/raw/`, `data/records.jsonl`, and `data/manifest.json`. Inference writes durable JSONL records under `results/raw/`; scoring, analysis, and router commands produce their corresponding subdirectories. Generated data and results are excluded from Git. Keep a separate backup of outputs needed for reproducibility. Dataset attribution and license notices are documented in [sources](docs/SOURCES.md).

The experiment fingerprint covers scientific Python files, configuration, dataset, and splits. A changed experiment requires a new results folder. The runner skips completed keys, records infrastructure failures separately, and stops on failures so the same command can resume after repair. Parsing failures and truncated generated outputs remain in the scored denominator. Back up and remove a partial final JSONL line before resuming; malformed logs are rejected.

Router training requires all development responses and seals development outputs, router checksums, and test IDs. Held-out inference is blocked until that freeze is valid; retraining after freeze is rejected. Use one writer per results directory. Local writer locks do not coordinate separate Colab runtimes.

## Source archive and Colab

```bash
python scripts/build_notebook.py
python scripts/package.py
```

Packaging creates `dist/indic-study.zip` with an `indic-study/` top-level directory containing source and supporting documents. Generated datasets, results, environments, and model caches are excluded. Open `notebooks/colab.ipynb` in Google Colab, choose a GPU runtime, and upload that archive. The notebook prepares the dataset, installs inference dependencies, and checkpoints results to Google Drive. Keep its configuration, runtime environment, and results folder consistent when resuming.
