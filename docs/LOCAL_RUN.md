# Local CUDA execution

`scripts/local_run.py` runs the preliminary workflow with `configs/local-nf4.json` and writes to `results/local-nf4-v1/`. It uses NF4 with double quantization and FP16 compute for both pinned Qwen checkpoints. The runner is intended for Linux with an NVIDIA CUDA GPU and requires the exact PyTorch build `2.8.0+cu126`.

## Environment and preparation

Use Python 3.12 and install Torch before the project extras:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv venv .venv-gpu --python 3.12
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv-gpu/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv-gpu/bin/python -e '.[analysis,gpu]' 'torch==2.8.0+cu126'

# Required even for the runner's preflight stage.
.venv-gpu/bin/python -m study --config configs/local-nf4.json prepare
.venv-gpu/bin/python -u scripts/local_run.py --stage preflight
```

Initial preparation and model downloads need internet access. Preflight checks CUDA availability, the exact Torch build, a CUDA tensor operation, GPU details through `nvidia-smi`, laptop mains power when available, RAM through `free`, disk space, and the experiment fingerprint. If the execution environment hides GPU devices, use a terminal with CUDA access. Connect laptop power before inference.

## Run and resume

From the repository root:

```bash
.venv-gpu/bin/python -u scripts/local_run.py

# Optional on Linux systems providing systemd-inhibit:
systemd-inhibit --what=sleep:idle --mode=block --why='Indic reasoning GPU study' .venv-gpu/bin/python -u scripts/local_run.py
```

The workflow completes one 23-condition problem per model, expands to 460 smoke responses, then 2,300 balanced pilot responses. Longest budgets run first within each problem. If the larger model fails specifically because of GPU memory, the runner records a fallback and completes all 100 development IDs on the smaller model. Other failures stop execution for diagnosis.

Use `--stage preflight`, `--stage feasibility`, `--stage smoke`, or `--stage pilot` to stop at that stage; each stage includes the preceding checks and work. The default `--stage all` reaches the same preliminary endpoint as `pilot`. Rerun the same command to preserve completed responses. Use one process at a time and keep scientific source, configuration, and environment unchanged after generating responses. Model downloads can resume from cached partial files.

The results directory contains an execution log, hardware/software preflight, run state, aggregate metrics, raw responses, scored records, balanced analysis, and a review sample. These outputs are ignored by Git; back them up separately. This runner does not train a router or perform held-out inference.

For other GPUs, PyTorch builds, precision choices, or result locations, use the `python -m study` commands described in the [README](../README.md). Keep configurations in separate results folders, and complete development plus router freeze before requesting held-out inference. Native-language script checks remain proxies; check parsing, truncation, and step compliance before interpreting accuracy differences.
