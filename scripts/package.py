"""Create the code-only source archive used by the Colab notebook."""
from pathlib import Path
import argparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]

# Exact file names keep datasets, generated outputs and local metadata out of
# the bundle even when they are present in the working directory.
SOURCE_FILES = (
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    "configs/study.json",
    "configs/local-nf4.json",
    "study/__init__.py",
    "study/__main__.py",
    "study/analysis.py",
    "study/characterization.py",
    "study/cli.py",
    "study/common.py",
    "study/data.py",
    "study/inference.py",
    "study/prompts.py",
    "study/results.py",
    "study/router.py",
    "study/scoring.py",
    "tests/test_pipeline.py",
    "tests/test_router.py",
    "tests/test_inference.py",
    "scripts/local_run.py",
    "scripts/package.py",
    "scripts/build_notebook.py",
    "notebooks/colab.ipynb",
    "docs/LOCAL_RUN.md",
    "docs/PROTOCOL.md",
    "docs/SOURCES.md",
    "docs/LICENSE.gsm8k.txt",
    "docs/LICENSE.mgsm.txt",
)


def source_paths():
    missing = [name for name in SOURCE_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Required source files are missing: " + ", ".join(missing)
        )
    return [ROOT / name for name in SOURCE_FILES]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    paths = source_paths()
    output_dir = ROOT / "dist"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "indic-study.zip"
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, Path("indic-study") / path.relative_to(ROOT))
    print(f"Created {output_path} ({len(paths)} source files)")


if __name__ == "__main__":
    main()
