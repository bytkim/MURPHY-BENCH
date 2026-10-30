"""Validated, exclusive writes for the single-writer pipeline."""
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

CONTRACTS = Path(__file__).resolve().parents[1] / "contract"


def strict_loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    # Also reject overflow such as 1e999, which parse_constant does not see.
    def finite(value):
        if isinstance(value, str):
            value.encode("utf-8")  # Reject lone surrogates before they break evidence writes.
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite JSON number")
        if isinstance(value, dict):
            for key, item in value.items():
                finite(key)
                finite(item)
        if isinstance(value, list):
            for item in value:
                finite(item)
    finite(result)
    return result


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def contract_identity(name):
    base, separator, version = name.rpartition('.v')
    if not separator:
        base, version = name, None
    base = {'comparison': 'util/comparison', 'receipt': 'util/receipt',
            'util/target': 'target'}.get(base, base)
    if base in ('grade', 'manifest', 'policy', 'response', 'run', 'target', 'task'):
        base = 'core/' + base
    return base, version


@lru_cache
def validator(name):
    resources = []
    for path in CONTRACTS.rglob("*.schema.json"):
        schema = strict_loads(path.read_text())
        Draft202012Validator.check_schema(schema)
        resources.append((path.as_uri(), Resource.from_contents(schema)))
    registry = Registry().with_resources(resources)
    base, version = contract_identity(name)
    folder = CONTRACTS / base
    if base == 'util/comparison' and version is None:
        # Keep the version-neutral reader without a second schema/alias file.
        schema = {'oneOf': [{'$ref': (folder / f'comparison-v{v}.schema.json').as_uri()}
                            for v in (1, 2)]}
    else:
        path = folder / f"{folder.name}-v{version or '1'}.schema.json"
        schema = {**strict_loads(path.read_text()), "$id": path.as_uri()}
    return Draft202012Validator(schema, registry=registry)


def validate(value, name):
    # Encoding also rejects nonfinite floats in otherwise unconstrained raw fields.
    encoded(value)
    validator(name).validate(value)
    if contract_identity(name)[0] == 'util/comparison' and value["schema_version"] == "2":
        # JSON Schema checks shape/types; fraction arithmetic needs a cross-field check.
        for measurement in value["measurements"].values():
            metrics = [measurement] if "status" in measurement else measurement.values()
            for metric in metrics:
                numerator, denominator = metric["numerator"], metric["denominator"]
                if denominator is None or denominator == 0:
                    valid = numerator == 0 and metric["value"] is None and metric["status"] == "unavailable"
                else:
                    valid = (numerator <= denominator and metric["status"] != "unavailable"
                             and metric["value"] == numerator / denominator)
                if not valid:
                    raise ValueError("Inconsistent measurement fraction/status")
    return value


def read_json(path, name):
    return validate(strict_loads(Path(path).read_text()), name)


def read_jsonl(path, name):
    with Path(path).open() as handle:
        return [validate(strict_loads(line), name) for line in handle]


def write_json(path, value, name):
    payload = encoded(validate(value, name))
    with Path(path).open("xb") as handle:
        handle.write(payload)


class JsonlWriter:
    def __init__(self, path, name):
        self.name = name
        self.handle = Path(path).open("xb")

    def __enter__(self):
        return self

    def append(self, value):
        payload = encoded(validate(value, self.name))
        self.handle.write(payload)
        self.handle.flush()

    def __exit__(self, *args):
        self.handle.close()


def load_build(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json", "manifest")
    policy = manifest["source"].get("policy")
    if policy is not None:
        snapshot = policy["snapshot"]
        if (digest(snapshot) != policy["snapshot_sha256"]
                or any(policy[k] != snapshot[k] for k in ("policy_id", "revision"))):
            raise ValueError("Policy snapshot identity mismatch")
    tasks = read_jsonl(path / "tasks.jsonl", "task")
    ids = {task["task_id"] for task in tasks}
    if len(ids) != len(tasks) or len(tasks) != manifest["task_count"]:
        raise ValueError("Duplicate task IDs or task-count mismatch")
    if file_hash(path / "tasks.jsonl") != manifest["tasks_sha256"]:
        raise ValueError("Task content hash mismatch")
    conditions = manifest["conditions"]
    if len({c["id"] for c in conditions}) != len(conditions):
        raise ValueError("Duplicate condition IDs")
    for condition in conditions:
        if not set(condition["task_ids"]) <= ids:
            raise ValueError("Condition refers to unknown tasks")
        if condition["reasoning"] == "on" and condition["endpoint"] != "/v1/chat/completions":
            raise ValueError("Reasoning requires chat")
        if "prompt_wrapper" in condition and condition["endpoint"] != "/completion":
            raise ValueError("Prompt wrapper is only valid for native completion")
    return manifest, {task["task_id"]: task for task in tasks}
