import json
from pathlib import Path
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator, ValidationError

from adapters.util.fixture import build
from murphy.util.io import CONTRACTS, JsonlWriter, load_build, read_json, strict_loads, validate, write_json


def test_schemas_are_valid():
    for path in CONTRACTS.rglob("*.schema.json"):
        Draft202012Validator.check_schema(json.loads(path.read_text()))




def test_fixture_contract(tmp_path):
    path = build(tmp_path)
    manifest, tasks = load_build(path)
    assert len(tasks) == manifest["task_count"] == 10
    assert len(manifest["conditions"]) == 6
    assert manifest["sampling"]["concurrency"] == 1
    with pytest.raises(FileExistsError):
        build(tmp_path)
    with (path / "tasks.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValidationError):
        load_build(path)


def test_reject_before_write(tmp_path):
    path = tmp_path / "invalid.json"
    with pytest.raises(ValidationError):
        write_json(path, {}, "task")
    assert not path.exists()
    with JsonlWriter(tmp_path / "rows.jsonl", "task") as writer:
        with pytest.raises(ValidationError):
            writer.append({})
    assert (tmp_path / "rows.jsonl").read_bytes() == b""


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}',
                                 '{"x":"\\ud800"}', '{"\\ud800":1}'])
def test_strict_json(text):
    with pytest.raises(ValueError):
        strict_loads(text)


def test_unknown_normalizer_and_extra_fields_rejected():
    with pytest.raises(ValidationError):
        validate({"mode": "strip", "unicode_normalize": True}, "util/normalizer")


def test_target_only_loopback():
    schema = json.loads((CONTRACTS / "core/target/target-v1.schema.json").read_text())
    validator = Draft202012Validator(schema["properties"]["base_url"])
    validator.validate("http://127.0.0.1:12345")
    with pytest.raises(ValidationError):
        validator.validate("https://external.example")
