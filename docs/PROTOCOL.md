# Experiment and analysis protocol

## Questions and treatment

RQ1: how does mathematical accuracy vary under requested 0/2/4/8/16-step explanations? RQ2: do these curves differ across English/Bengali/Telugu and the two model sizes? RQ3: can a question-only router reduce generated tokens while retaining accuracy?

The primary treatment is requested visible explanation length. Qwen native thinking is disabled. Actual steps/tokens are secondary measurements; they are not evidence of hidden thought length. A no-cliff or unsuccessful-router result remains valid.

## Data, split, and workload

Use original MGSM parallel rows from Google Research revision `3622039cf51f7eeffa58b957332a8e8c337981d7`. IDs are `mgsm-000` through `mgsm-249` in original row order. Preparation saves source files, SHA-256 values, numeric alignment checks, and the record checksum under `data/`. Matching answers alone do not prove translation quality; semantic quality relies on the published benchmark. See [sources and licenses](SOURCES.md).

Shuffle IDs with Python's `random.Random(42)`: first 100 development, remaining 150 test. Pilot is first 50 development IDs; smoke is first 10. Never split translations separately. Models and generation settings are in `configs/study.json`; checkpoint revisions are pinned.

Each model/problem has 15 native conditions (three languages × five budgets) and eight pivot conditions (two Indic languages × four nonzero budgets). English pivot and zero-step pivot are omitted as redundant. Total: 11,500 responses; development: 4,600; test: 6,900; balanced pilot: 2,300. Infrastructure failures do not count; resume after addressing their cause. Parse failures/truncated generated responses remain in the evaluation denominator and are flagged.

## Output and scoring

The same English instruction template is used for all languages. Source questions remain untranslated. Explanations and final answer sentences use the requested languages; the last line is a common numeric marker. The scorer accepts exactly one standalone `FINAL_ANSWER:` line and one finite decimal number. Normalize Unicode decimal digits, minus signs, valid western/Indian digit grouping, and equivalent decimal representations. Never infer a final answer from intermediate numbers. Missing, multiple, ambiguous, and malformed final answers are incorrect.

Count numbered explanation lines and exact repeated lines. Step compliance means numbered-line count equals the request; it does not prove semantic step quality or detect all unnumbered reasoning. Script diagnostics inspect letters in numbered lines, exclude the final marker, and are explicitly proxies. These automated checks do not establish native-language fluency or semantic correctness.

## Reproducibility and memory

Batch size 1, FP16, sampling enabled, temperature 0.7, top-p 0.8, top-k 20, common maximum 4,096 new tokens. Per-condition seeds are derived deterministically from the configuration and experiment key. Store versions, device, latency, peak memory, token counts, model revision, actual prompt, termination state, and raw text. Sampling need not be bit-identical across different GPU/library environments.

Start both models at the longest budget during smoke evaluation. If insufficient GPU memory requires 4-bit NF4, set it globally for both models before official results and use a separate results directory for that configuration. Do not pool configurations. Truncation and step failures must be discussed before interpreting any apparent decline.

`configs/local-nf4.json` selects uniform NF4 weight quantization with double quantization and FP16 compute for both checkpoints; other generation settings match the main configuration. The local runner requires Torch 2.8.0+cu126 and records software/hardware preflight information. Longest-first ordering preserves complete model/problem blocks and experiment IDs.

## Quantitative analysis

Curves use only complete matched problem blocks across the selected models and conditions; omitted partial blocks are reflected in coverage. A one-model fallback must be explicitly requested with `--model`. Report per-condition accuracy, actual steps, generated tokens, numbered-step compliance, parse failure, and truncation.

Predefined comparisons: native 16 minus 4 steps within each model/language; the Bengali/English and Telugu/English differences in that effect; pivot minus native at each nonzero budget. Bootstrap shared problem IDs with 2,000 resamples and seed 42. Intervals are exploratory 95% intervals, without multiplicity-adjusted significance claims. Do not call a noisy peak or isolated negative contrast a demonstrated capacity cliff. Describe model-size associations, not a causal effect of parameter count alone.

## Router and baselines

Fit one native-budget router per model. Features: language, Unicode character length, punctuation-based sentence count, and digit-based quantity count. No gold, model answer, English translation, or generated token count is available at prediction time. Digit counts miss quantities spelled as words; record this limitation.

Labels: the smallest budget with a correct development response. Exclude and report questions with no successful budget; do not invent a successful label. With no successful labels at all, report a language-lookup fallback instead of a trained tree. A single sample per condition makes labels noisy.

Decision tree depth 3, minimum leaf size 10, random state 42; DictVectorizer encodes language. Five-fold GroupKFold holds all translations of a problem together. Fit router and all development-selected baselines inside each training fold. Baselines: always eight; dev-best global per model; dev-best per language/model, with shortest-budget tie breaking. Fit an otherwise identical router without language as the ablation. Report both accuracy and mean output tokens; no improvement is assumed.

Once all 4,600 development outputs are complete, fit on full development and freeze router artifact checksums, code/config/data fingerprints, development-output checksum, and test IDs. Held-out inference is locked until this freeze. Retraining is rejected after freeze. Evaluate every test question using the frozen policy and cached native outputs; this requires no additional generations. Bootstrap problem IDs, keeping all three translations together, for router/baseline paired accuracy and token differences.

## Human review and limitations

Sample up to 40 incorrect outputs, balanced by model/question language/explanation language. English explanations can be checked against the original parallel English question. Without a qualified native-language reviewer, native explanations permit only observed script/repetition/count checks; use `unassessable_native` for semantic uncertainty. Have a second reviewer independently review at least 10 English cases.

Categories: initial arithmetic, repetition, contradiction, changed correct intermediate result, step noncompliance, script mixing, unassessable native, other. Explain sample selection and limitations; no LLM-as-judge is used. Numerical accuracy, requested-line compliance, and script proportions must remain distinct from semantic explanation quality. Report actual response coverage and incomplete stages alongside findings.
