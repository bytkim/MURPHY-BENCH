"""Run the public test fixture through the real pipeline, with mocked HTTP."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from pytest import MonkeyPatch
from test_pipeline import make_pipeline
from test_experiment import completed
from scripts.report import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with MonkeyPatch.context() as patch:
        completion, _ = completed(make_pipeline(output, patch, chat_usage=True), output)
        destination = report(completion, output / "report")
    print(destination)


if __name__ == "__main__":
    main()
