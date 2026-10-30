"""Task grades remain separate from token/probability-record integrity."""
import base64
from pathlib import Path

from murphy.client import joined, parse_body
from murphy.core.run import check_properties, request_for
from murphy.util.io import JsonlWriter, digest, file_hash, load_build, read_json, read_jsonl
from murphy.verifiers.exact_match import verify


def key(response):
    return response["task_id"], response["condition"]["id"], response["mode"], response["repeat"]


def load_run(build_dir, run_dir):
    build_dir, run_dir = Path(build_dir), Path(run_dir)
    manifest, tasks = load_build(build_dir)
    meta = read_json(run_dir / "run.json", "run")
    if meta["build_id"] != manifest["build_id"] or meta["manifest_sha256"] != file_hash(build_dir / "manifest.json"):
        raise ValueError("Run does not belong to this immutable build")
    target = read_json(run_dir / "target.json", "target")
    if target["target_id"] != meta["target_id"] or any(target[k] != meta["fingerprint"][k] for k in
            ("model_sha256", "binary_sha256", "source_commit", "runtime_config")):
        raise ValueError("Saved target disagrees with run fingerprint")
    check_properties(target, meta["fingerprint"]["server_props"])
    responses = read_jsonl(run_dir / "responses.jsonl", "response")
    conditions = {c["id"]: c for c in manifest["conditions"]}
    expected = {(t, c["id"], m, r) for c in manifest["conditions"] for t in c["task_ids"]
                for m in manifest["modes"] for r in range(manifest["repeats"])}
    actual = [key(r) for r in responses]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Missing, duplicate, or unexpected attempt rows")
    for response in responses:
        if any(response[k] != meta[k] for k in ("build_id", "run_id", "target_id")):
            raise ValueError("Response identity mismatch")
        condition = conditions[response["condition"]["id"]]
        if response["condition"] != condition or response["request"] != request_for(tasks[response["task_id"]], condition, response["mode"]):
            raise ValueError("Response request/condition mismatch")
        captured = parse_body(base64.b64decode(response["body_base64"], validate=True), condition["endpoint"], response["mode"] == "stream")
        if any(response[k] != captured[k] for k in ("events", "chunks")):
            raise ValueError("Parsed observations disagree with preserved response bytes")
        if response["status"] in ("complete", "incomplete", "parse_error"):
            if response["http_status"] != 200 or any(response[k] != captured[k] for k in ("status", "error")):
                raise ValueError("Response completion status disagrees with preserved bytes or HTTP status")
        elif not response["error"]:
            raise ValueError("Transport failure requires an error observation")
        if response["status"] == "http_error":
            if response["http_status"] in (None, 200):
                raise ValueError("HTTP failure requires an unsuccessful HTTP status")
        elif response["http_status"] not in (None, 200):
            raise ValueError("Unsuccessful HTTP status requires http_error")
    return meta, tasks, {key(r): r for r in responses}


def grade_one(response, task):
    if response["status"] != "complete":
        passed, reason = None, "incomplete_transport"
    elif not response["condition"]["grade"]:
        passed, reason = None, "intentional_condition"
    else:
        passed = verify(joined(response)["content"], task["expected"], task["verifier"]["settings"])
        reason = "exact_match" if passed else "answer_mismatch"
    return {"schema_version": "1", **{k: response[k] for k in ("build_id", "run_id", "target_id", "task_id", "mode", "repeat")},
            "condition_id": response["condition"]["id"], "response_sha256": digest(response), "pass": passed, "reason": reason}


def grade(build_dir, run_dir, output=None):
    _, tasks, responses = load_run(build_dir, run_dir)
    destination = Path(output) if output else Path(run_dir) / "grades.jsonl"
    with JsonlWriter(destination, "grade") as writer:
        for response in responses.values():
            writer.append(grade_one(response, tasks[response["task_id"]]))
    return destination
