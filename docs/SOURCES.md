# Benchmark provenance and licensing

MGSM questions and numerical targets come from the original [Google Research parallel release](https://github.com/google-research/url-nlp/tree/3622039cf51f7eeffa58b957332a8e8c337981d7/mgsm), pinned to revision `3622039cf51f7eeffa58b957332a8e8c337981d7`. The English, Bengali, and Telugu files each contain 250 question/answer pairs.

Run `python -m study prepare` to download the original TSV files into `data/raw/<revision>/`, normalize records into `data/records.jsonl`, and record original-source URLs, SHA-256 checksums, dataset checksum, and deterministic shared-ID splits in `data/manifest.json`. These generated files are excluded from Git. Preparation reuses existing source files and rejects a changed manifest rather than overwriting frozen splits.

The records are the published parallel benchmark. The project uses row-based IDs shared across translations and verifies numerical answers across all rows. Matching answers alone do not establish semantic translation alignment. Development and held-out membership are fixed in the generated manifest.

The MGSM license notice is retained in [LICENSE.mgsm.txt](LICENSE.mgsm.txt), copied from the adjacent license at the pinned source revision. MGSM uses Creative Commons Attribution 4.0 International. The underlying [GSM8K source](https://github.com/openai/grade-school-math) has its MIT license notice retained in [LICENSE.gsm8k.txt](LICENSE.gsm8k.txt). These notices describe the benchmark sources, rather than granting a new license for this project's code. The preparation command downloads data files and does not download license notices.

`python -m study characterize` optionally generates `data/characterization/` from the development IDs, recording tokenizer revisions, software versions, and methodology. Input-token measurements are separate from model-answer evaluation and do not count as completed inference responses. Dataset records contain no generated model answers.
