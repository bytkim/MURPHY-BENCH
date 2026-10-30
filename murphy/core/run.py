"""Sequential capture. No retries, repair, overwrites, or dataset branches."""
import json
import re
import tomllib
import uuid
from pathlib import Path

from murphy.client import Client, now
from murphy.util.io import JsonlWriter, file_hash, load_build, validate, write_json


def check_properties(target, props):
    build_info = props.get("build_info")
    revisions = re.findall(r"(?<![0-9a-f])[0-9a-f]{7,40}(?![0-9a-f])", build_info) if isinstance(build_info, str) else []
    if not any(target["source_commit"].startswith(revision) for revision in revisions):
        raise ValueError("Observed server build_info disagrees with target revision")
    if type(props.get("total_slots")) is not int or props["total_slots"] != 1:
        raise ValueError("Observed server must expose exactly one slot")


def request_for(task, condition, mode):
    request = {"temperature": 0, "seed": 42, "cache_prompt": False,
               "return_tokens": True, "stream": mode == "stream", "stop": condition["stop"]}
    if condition["literal_grammar"]:
        text = task["expected"]
        if condition["reasoning"] == "on":
            text = "<think>fixture</think>" + text
        request["grammar"] = "root ::= " + json.dumps(text, ensure_ascii=False)
    if condition["endpoint"] == "/completion":
        wrapper = condition.get("prompt_wrapper", {"prefix": "", "suffix": ""})
        request.update(prompt=wrapper["prefix"] + task["prompt"] + wrapper["suffix"],
                       n_predict=condition["max_tokens"], n_probs=5)
    else:
        request.update(messages=[{"role": "user", "content": task["prompt"]}],
                       max_tokens=condition["max_tokens"], logprobs=True, top_logprobs=5,
                       chat_template_kwargs={"enable_thinking": condition["reasoning"] == "on"})
        if mode == "stream":
            request["stream_options"] = {"include_usage": True}
    return request


def run(build_dir, target, *, client_factory=Client):
    build_dir = Path(build_dir)
    manifest, tasks = load_build(build_dir)
    if not isinstance(target, dict):
        with Path(target).open("rb") as handle:
            target = tomllib.load(handle)
    validate(target, "target")
    directory = build_dir / "runs" / target["target_id"]
    if directory.exists():
        raise FileExistsError(f"Target run already exists; preserve it and use an explicit new target ID: {directory}")
    for key in ("model", "binary"):
        if file_hash(target[f"{key}_path"]) != target[f"{key}_sha256"]:
            raise ValueError(f"Target {key} hash mismatch")
    if any(c["reasoning"] == "on" for c in manifest["conditions"]) and not target["reasoning_capable"]:
        raise ValueError("Reasoning condition requires a qualified template")
    with client_factory(target["base_url"]) as client:
        props = client.properties()
        check_properties(target, props)
        run_id = uuid.uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        fingerprint = {key: target[key] for key in ("model_sha256", "binary_sha256", "source_commit", "runtime_config")}
        fingerprint["server_props"] = props
        write_json(directory / "target.json", target, "target")
        write_json(directory / "run.json", {"schema_version": "1", "build_id": manifest["build_id"],
                   "run_id": run_id, "target_id": target["target_id"], "manifest_sha256": file_hash(build_dir / "manifest.json"),
                   "fingerprint": fingerprint, "created_at": now()}, "run")
        with JsonlWriter(directory / "responses.jsonl", "response") as writer:
            for condition in manifest["conditions"]:
                for task_id in condition["task_ids"]:
                    for mode in manifest["modes"]:
                        for repeat in range(manifest["repeats"]):
                            request = request_for(tasks[task_id], condition, mode)
                            result = client.complete(condition["endpoint"], request)
                            writer.append({"schema_version": "1", "build_id": manifest["build_id"],
                                "run_id": run_id, "target_id": target["target_id"], "task_id": task_id,
                                "condition": condition, "mode": mode, "repeat": repeat, "request": request, **result})
    return directory
