"""python -m murphy.cli {fixture,run,grade,compare}."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    fixture = stages.add_parser("fixture")
    fixture.add_argument("--output", default="data/dev")
    run = stages.add_parser("run")
    run.add_argument("build")
    run.add_argument("target")
    grade = stages.add_parser("grade")
    grade.add_argument("build")
    grade.add_argument("run")
    grade.add_argument("--output")
    compare = stages.add_parser("compare")
    compare.add_argument("build")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument("--output")
    args = parser.parse_args()
    if args.stage == "fixture":
        from adapters.util.fixture import build
        result = build(args.output)
    elif args.stage == "run":
        from murphy.core.run import run
        result = run(args.build, args.target)
    elif args.stage == "grade":
        from murphy.core.grade import grade
        result = grade(args.build, args.run, args.output)
    else:
        from murphy.core.compare import compare
        result = compare(args.build, args.left, args.right, args.output)
    print(result)


if __name__ == "__main__":
    main()
