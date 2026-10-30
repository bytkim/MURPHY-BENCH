"""Pair eligible runs without assuming either build is the correctness oracle."""
from pathlib import Path

from murphy.core.grade import grade_one, load_run
from murphy.core.measure import inspect, numerical, stability
from murphy.core.metrics import measurements
from murphy.util.io import JsonlWriter, file_hash, read_json
from murphy.util.vocabulary import Vocabulary


def compare(build_dir, left_dir, right_dir, output=None):
    lm, tasks, left = load_run(build_dir, left_dir)
    rm, _, right = load_run(build_dir, right_dir)
    a, b = lm["fingerprint"], rm["fingerprint"]
    if any(a[k] != b[k] for k in ("model_sha256", "runtime_config")):
        raise ValueError("Model or runtime configuration differs")
    if a["server_props"].get("chat_template") != b["server_props"].get("chat_template"):
        raise ValueError("Chat templates differ")
    target = read_json(Path(left_dir) / "target.json", "target")
    if file_hash(target["model_path"]) != a["model_sha256"]:
        raise ValueError("Vocabulary GGUF no longer matches the captured model")
    vocabulary = Vocabulary(target["model_path"])
    groups = sorted({k[:3] for k in left})
    flat = all(Path(directory).resolve() == (Path(build_dir) / 'runs' / meta['target_id']).resolve()
               for directory, meta in ((left_dir, lm), (right_dir, rm)))
    key = 'target_id' if flat else 'run_id'  # Historical nested attempts keep their names.
    destination = Path(output) if output else Path(build_dir) / "compare" / f'{lm[key]}__{rm[key]}.jsonl'
    destination.parent.mkdir(parents=True, exist_ok=True)
    def repeats(rows, group):
        return [row for key, row in sorted(rows.items()) if key[:3] == group]
    with JsonlWriter(destination, "comparison") as writer:
        for group in groups:
            task_id, condition_id, mode = group
            lrows, rrows = repeats(left, group), repeats(right, group)
            sides = []
            for rows, all_rows in ((lrows, left), (rrows, right)):
                reference = repeats(all_rows, (task_id, condition_id, "nonstream"))
                sides.append({"stable_repeats": stability(rows), "complete_responses": sum(r["status"] == "complete" for r in rows),
                              "complete_response_rate": sum(r["status"] == "complete" for r in rows) / len(rows),
                              "attempts": len(rows), "grades": [grade_one(r, tasks[task_id]) for r in rows],
                              "checks_by_repeat": [inspect(r, reference, vocabulary) for r in rows]})
            numeric = numerical(lrows, rrows)
            writer.append({"schema_version": "2", "build_id": lm["build_id"], "left_run_id": lm["run_id"],
                           "right_run_id": rm["run_id"], "task_id": task_id, "condition_id": condition_id,
                           "mode": mode, "left": sides[0], "right": sides[1], "numerical": numeric,
                           "references": {"model_sha256": a["model_sha256"], "vocabulary": vocabulary.identity,
                                          "left_nonstream_run_id": lm["run_id"], "right_nonstream_run_id": rm["run_id"]},
                           "measurements": measurements(lrows, rrows, sides[0]["grades"], sides[1]["grades"], numeric)})
    return destination
