"""Run the local NF4 smoke and pilot workflow with durable execution summaries."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from study.common import context_hash, load_config, read_json, read_jsonl, write_json
from study.data import load_data
from study.inference import environment
from study.results import coverage, load_scored

CONFIG = ROOT / 'configs/local-nf4.json'
RESULTS = ROOT / 'results/local-nf4-v1'


def state_data():
    cfg = load_config(CONFIG)
    manifest, records = load_data(cfg, ROOT / 'data')
    rows = load_scored(cfg, manifest, records, RESULTS)
    return cfg, manifest, records, rows


def cli(*arguments):
    command = [sys.executable, '-u', '-m', 'study', '--config', str(CONFIG),
               '--results-dir', str(RESULTS), *arguments]
    with (RESULTS / 'execution.log').open('a', buffering=1) as log:
        print('Running:', ' '.join(arguments), flush=True)
        log.write('\n' + datetime.now(timezone.utc).isoformat() + ' ' + ' '.join(arguments) + '\n')
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def preflight():
    import torch
    assert torch.cuda.is_available(), 'CUDA unavailable: run outside the sandbox.'
    assert torch.__version__ == '2.8.0+cu126', f'Unexpected Torch build: {torch.__version__}'
    value = torch.ones(8, device='cuda').sum().item()
    assert value == 8
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version,temperature.gpu',
                                   '--format=csv'], text=True)
    supplies = []
    for path in Path('/sys/class/power_supply').glob('*'):
        if (path / 'type').exists() and (path / 'type').read_text().strip() == 'Mains':
            supplies.append((path / 'online').read_text().strip() == '1')
    if supplies and not any(supplies):
        raise RuntimeError('Connect laptop power before starting this run.')
    cfg, manifest, _, _ = state_data()
    payload = {'checked_at': datetime.now(timezone.utc).isoformat(),
               'config_hash': context_hash(cfg, manifest), 'python': sys.version,
               'versions': environment(), 'cuda_runtime': torch.version.cuda,
               'gpu': gpu, 'cuda_tensor_check': value, 'mains_online': supplies,
               'memory': subprocess.check_output(['free', '-h'], text=True),
               'disk_free_bytes': shutil.disk_usage(ROOT).free}
    write_json(RESULTS / 'preflight.json', payload)
    print(gpu, flush=True)


def handoff(stage, target):
    cfg, manifest, records, rows = state_data()
    metrics = []
    for model in cfg['models']:
        selected = [row for row in rows if row['model'] == model]
        if not selected:
            continue
        count = len(selected)
        mean = sum(row['latency_seconds'] for row in selected) / count
        metrics.append({'model': model, 'completed': count,
                        'accuracy': sum(row['correct'] for row in selected) / count,
                        'mean_seconds': mean,
                        'total_generation_seconds': sum(row['latency_seconds'] for row in selected),
                        'mean_output_tokens': sum(row['output_tokens'] for row in selected) / count,
                        'max_output_tokens': max(row['output_tokens'] for row in selected),
                        'peak_allocated_bytes': max(row['peak_gpu_memory_bytes'] for row in selected),
                        'peak_reserved_bytes': max(row['peak_gpu_reserved_bytes'] for row in selected),
                        'parse_failures': sum(row['parse_error'] is not None for row in selected),
                        'truncated': sum(row['truncated'] for row in selected),
                        'step_compliance': sum(row['step_compliant'] for row in selected) / count,
                        'remaining_target_hours_with_30_percent_overhead': mean * max(0, (2300 if target == 'smaller-model-development' and model == 'qwen3-1.7b' else 1150)-count) * 1.3 / 3600})
    report = {'stage': stage, 'target': target, 'config_hash': context_hash(cfg, manifest),
              'recorded_at': datetime.now(timezone.utc).isoformat(),
              'coverage': coverage(cfg, manifest, records, rows), 'models': metrics,
              'errors': {model: read_jsonl(RESULTS / f'raw/{model}.errors.jsonl') for model in cfg['models']}}
    write_json(RESULTS / 'execution_report.json', report)
    lines = ["# Local execution summary", '', f'Stage: {stage}. Target: {target}.',
             '', f'Recorded responses: {len(rows)}; configuration: `{report["config_hash"]}`.', '',
             '| Model | Responses | Mean seconds | Peak allocated GiB | Mean tokens | Parse failures | Truncated | Step compliance |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for metric in metrics:
        lines.append(f'| {metric["model"]} | {metric["completed"]} | {metric["mean_seconds"]:.2f} | {metric["peak_allocated_bytes"]/2**30:.2f} | {metric["mean_output_tokens"]:.1f} | {metric["parse_failures"]} | {metric["truncated"]} | {metric["step_compliance"]:.1%} |')
    completed_models = ', '.join(cfg['models'][metric['model']]['name'] for metric in metrics) or 'no models yet'
    lines += ['', '## Execution method', '',
              f'This stage contains {len(rows)} completed responses from {completed_models}. '
              'The local configuration uses bitsandbytes NF4 weight quantization with double quantization '
              'and FP16 compute, batch size one, native thinking disabled, and the pinned sampling settings. '
              'The preflight and raw response metadata record the actual device and software environment. '
              'Longest requested budgets run first within each matched problem block. Completed responses '
              'are flushed to disk immediately, and subsequent stages reuse the same experiment IDs. '
              'Requested explanation length is not a measurement of hidden reasoning.', '',
              '## Interpretation', '',
              'Check coverage, parsing, truncation, and numbered-step compliance before interpreting curves. '
              'Script diagnostics are proxies and do not establish native-language fluency. '
              'Router training and held-out inference require the later development and freeze workflow. '
              'The JSON report includes exact phase coverage and infrastructure errors; partial blocks '
              'do not count as completed milestones.', '', '## Available artifacts', '']
    artifacts = [
        ('execution_report.json', 'Execution report and metrics'),
        ('preflight.json', 'Hardware and software preflight'),
        ('execution.log', 'Execution log'),
        ('scored/development.jsonl', 'Scored development records'),
        ('analysis/development_review.csv', 'Incorrect-response review sample'),
    ]
    for phase in ['smoke', 'pilot', 'development']:
        for model in ['both-models', *cfg['models']]:
            artifacts.append((f'analysis/{phase}/{model}/report.md', f'{phase} analysis: {model}'))
    for model in cfg['models']:
        artifacts.append((f'raw/{model}.jsonl', f'Raw responses: {model}'))
    for relative_path, label in artifacts:
        if (RESULTS / relative_path).exists():
            lines.append(f'- [{label}]({relative_path})')
    if target == 'smaller-model-development':
        lines += ['', '**Fallback:** complete all 100 development IDs for the smaller model. '
                  'The two-model comparison remains incomplete; larger-model attempt counts stay separate.']
    (RESULTS / 'RUN_SUMMARY.md').write_text('\n'.join(lines) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['preflight', 'feasibility', 'smoke', 'pilot', 'all'], default='all')
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
    state_path = RESULTS / 'run_state.json'
    cfg, manifest, _, _ = state_data()
    fingerprint = context_hash(cfg, manifest)
    state = read_json(state_path) if state_path.exists() else {'config_hash': fingerprint, 'target': 'balanced-pilot'}
    assert state['config_hash'] == fingerprint, 'Run fingerprint changed.'
    preflight()
    if args.stage == 'preflight':
        return
    write_json(state_path, state)
    try:
        first_id = manifest['splits']['smoke'][0]
        for model in cfg['models']:
            if model == 'qwen3-4b' and state['target'] != 'balanced-pilot':
                continue
            _, _, _, rows = state_data()
            done = sum(row['model'] == model and row['problem_id'] == first_id for row in rows)
            if done < 23:
                try:
                    cli('run', '--phase', 'smoke', '--model', model, '--limit', str(23-done))
                except subprocess.CalledProcessError:
                    errors = read_jsonl(RESULTS / f'raw/{model}.errors.jsonl')
                    memory_failure = errors and any(term in errors[-1]['error'].lower() for term in ['out of memory', 'offloaded to cpu/disk'])
                    if model != 'qwen3-4b' or not memory_failure:
                        raise
                    state['target'] = 'smaller-model-development'
                    write_json(state_path, state)
                    print('Larger model memory failure: using documented smaller-model fallback.', flush=True)
        cli('score', '--split', 'development')
        handoff('feasibility', state['target'])
        if args.stage == 'feasibility':
            return
        options = ['--model', 'qwen3-1.7b'] if state['target'] != 'balanced-pilot' else []
        cli('run', '--phase', 'smoke', *options)
        cli('analyze', '--phase', 'smoke', *options)
        report = handoff('smoke', state['target'])
        print(json.dumps(report['models'], indent=2), flush=True)
        if args.stage == 'smoke':
            return
        phase = 'pilot' if state['target'] == 'balanced-pilot' else 'development'
        cli('run', '--phase', phase, *options)
        cli('score', '--split', 'development')
        cli('analyze', '--phase', phase, *options)
        cli('review-sample', '--split', 'development')
        cli('status')
        report = handoff('completed', state['target'])
        if phase == 'pilot':
            assert report['coverage']['pilot']['complete'] == 2300
            assert report['coverage']['pilot']['complete_paired_ids'] == 50
        else:
            _, _, _, rows = state_data()
            assert sum(row['model'] == 'qwen3-1.7b' for row in rows) == 2300
        print('Local pilot workflow completed. Held-out inference remains locked.', flush=True)
    except BaseException:
        handoff('interrupted_or_failed', state['target'])
        raise


if __name__ == '__main__':
    main()
