"""Prepare immutable dataset inputs from admitted local sources and model metadata."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.core.prepare import prepare, parse_prompt_revisions
from adapters.core.policy import DEFAULT as DEFAULT_POLICY

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Public completions + chat experiment: explicitly select --policy policies/murphy.v4.json --profile full. "
               "Endpoints become matched manifest conditions; modes and repeats remain manifest settings. "
               "Preparation does not launch evaluation.")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY,
                        help="Explicit recipe file; default is historical native-only murphy.v1.json, not latest")
    parser.add_argument("--profile", choices=("trial", "full"), default="trial")
    parser.add_argument("--prompt-revision", action="append", default=[], metavar="NAME=N",
                        help="Compatibility assertion: must match the selected policy's prompt revision.")
    args = parser.parse_args()
    print(prepare(args.sources, args.model_metadata, args.output, policy_path=args.policy, profile=args.profile,
                  prompt_revisions=parse_prompt_revisions(args.prompt_revision)))
