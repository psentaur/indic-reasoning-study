"""Single-GPU, resumable Hugging Face inference. Imported only for actual runs."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import importlib.metadata
import tempfile
import time
from pathlib import Path

from .common import append_jsonl, context_hash, digest, read_json, write_json
from .prompts import build_grid
from .results import load_scored, verify_freeze


@contextmanager
def exclusive_writer(path: Path):
    # Use local POSIX locks: mounted Drive filesystems need not implement flock.
    # One Colab runtime owns a results directory; this is not a distributed lock.
    lock_directory = Path(tempfile.gettempdir()) / "indic-study-writer-locks"
    lock_directory.mkdir(exist_ok=True)
    local_lock = lock_directory / (digest(str(path.resolve())) + ".lock")
    with local_lock.open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another runner is already writing to this results folder") from error
        yield


def environment() -> dict:
    return {name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "accelerate", "bitsandbytes", "tokenizers", "jinja2", "safetensors", "huggingface-hub"]}


def execution_order(rows: list[dict]) -> list[dict]:
    """Keep model/problem blocks intact; stress longest budgets first."""
    blocks = {}
    for row in rows:
        blocks.setdefault((row["model"], row["problem_id"]), []).append(row)
    return [row for block in blocks.values() for row in sorted(block, key=lambda item: -item["budget"])]


@contextmanager
def log_failure(results_dir: Path, model_key: str, stage: str, **details):
    try:
        yield
    except Exception as error:
        append_jsonl(results_dir / f"raw/{model_key}.errors.jsonl",
                     {**details, "stage": stage, "error_type": type(error).__name__,
                      "error": str(error), "time": datetime.now(timezone.utc).isoformat()})
        raise


def run(config: dict, manifest: dict, records: list[dict], results_dir: Path,
        phase: str, model_filter: str | None = None, limit: int | None = None) -> dict:
    current_hash = context_hash(config, manifest)
    if phase == "test":
        verify_freeze(results_dir / "freeze.json", current_hash)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA GPU is required. Run outside the device-isolated sandbox on your local GPU, or use notebooks/colab.ipynb.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    versions = environment()
    grid = build_grid(config, manifest, records, phase, current_hash, model_filter)
    generation = config["generation"]
    started = time.perf_counter()
    completed = 0
    with exclusive_writer(results_dir / "raw/.writer.lock"):
        # Validate even other phases/models; refuse mixing stale files.
        existing = load_scored(config, manifest, records, results_dir)
        seen = {row["experiment_id"] for row in existing}
        metadata_path = results_dir / "raw/environment.json"
        metadata = {"config_hash": current_hash, "versions": versions}
        if metadata_path.exists() and read_json(metadata_path) != metadata:
            raise ValueError("Software versions or experiment changed. Restore the original environment or use a new results directory.")
        write_json(metadata_path, metadata)
        pending = execution_order([row for row in grid if row["experiment_id"] not in seen])
        if limit:
            pending = pending[:limit]
        print(f"{len(grid)} planned; {len(grid) - len([r for r in grid if r['experiment_id'] not in seen])} already complete; {len(pending)} to run", flush=True)
        for model_key, specification in config["models"].items():
            jobs = [row for row in pending if row["model"] == model_key]
            if not jobs:
                continue
            with log_failure(results_dir, model_key, "model_loading", config_hash=current_hash):
                tokenizer = AutoTokenizer.from_pretrained(specification["name"], revision=specification["revision"])
                kwargs = {"revision": specification["revision"], "torch_dtype": torch.float16, "device_map": "auto"}
                if generation["quantization"] == "4bit":
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                      bnb_4bit_compute_dtype=torch.float16,
                                                                      bnb_4bit_use_double_quant=True)
                model = AutoModelForCausalLM.from_pretrained(specification["name"], **kwargs).eval()
                if any(str(device) in {"cpu", "disk"} for device in getattr(model, "hf_device_map", {}).values()):
                    raise RuntimeError("Model was offloaded to CPU/disk. Select 4bit for BOTH models before official runs, or obtain sufficient GPU memory.")
            for job in jobs:
                set_seed(job["seed"])
                text = tokenizer.apply_chat_template([{"role": "user", "content": job["prompt"]}],
                                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                inference_start = time.perf_counter()
                try:
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=generation["max_new_tokens"],
                                                   do_sample=generation["do_sample"], temperature=generation["temperature"],
                                                   top_p=generation["top_p"], top_k=generation["top_k"],
                                                   pad_token_id=tokenizer.eos_token_id)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - inference_start
                    output_ids = generated[0, inputs["input_ids"].shape[-1]:]
                    response = tokenizer.decode(output_ids, skip_special_tokens=True)
                    eos_ids = model.generation_config.eos_token_id
                    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
                    ended_with_eos = len(output_ids) > 0 and int(output_ids[-1]) in eos_ids
                    truncated = len(output_ids) >= generation["max_new_tokens"] and not ended_with_eos
                    row = {**job, "schema_version": 1, "backend": "transformers", "status": "complete",
                           "model_name": specification["name"], "model_revision": specification["revision"],
                           "response": response, "input_tokens": inputs["input_ids"].shape[-1],
                           "output_tokens": len(output_ids), "latency_seconds": elapsed,
                           "truncated": truncated, "finish_reason": "length" if truncated else "eos",
                           "gpu_name": torch.cuda.get_device_name(),
                           "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                           "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                           "environment": versions, "generated_at": datetime.now(timezone.utc).isoformat()}
                    append_jsonl(results_dir / f"raw/{model_key}.jsonl", row)
                    completed += 1
                    if completed == 1 or completed % 10 == 0:
                        print(f"Saved {completed}/{len(pending)} new responses; last took {elapsed:.1f}s", flush=True)
                except Exception as error:
                    append_jsonl(results_dir / f"raw/{model_key}.errors.jsonl",
                                 {"experiment_id": job["experiment_id"], "error_type": type(error).__name__,
                                  "error": str(error), "time": datetime.now(timezone.utc).isoformat()})
                    # Stop on failure; completed responses remain durable and this key can be retried.
                    raise
                del inputs, generated, output_ids
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    return {"new_responses": completed, "wall_seconds": time.perf_counter() - started,
            "seconds_per_new_response": (time.perf_counter() - started) / completed if completed else None}
