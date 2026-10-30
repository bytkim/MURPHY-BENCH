import json
from copy import deepcopy

import httpx
import pytest

from adapters.util.fixture import build
from conftest import ByteVocabulary, native, record, sse
from murphy.client import Client
from murphy.core.compare import compare
from murphy.core.grade import grade, load_run
from murphy.core.run import check_properties, run
from murphy.util.io import file_hash, read_jsonl, validate
from jsonschema import ValidationError


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    return make_pipeline(tmp_path, monkeypatch)


def make_pipeline(tmp_path, monkeypatch, *, chat_usage=False):
    build_dir = build(tmp_path)
    model, binary = tmp_path / "model", tmp_path / "binary"
    model.write_bytes(b"unit-test model identity; not a GGUF")
    binary.write_bytes(b"unit-test binary identity; not an executable")
    target = {"schema_version": "1", "target_id": "mock-patched", "base_url": "http://127.0.0.1:1",
              "model_path": str(model), "binary_path": str(binary), "source_commit": "a" * 40,
              "model_sha256": file_hash(model), "binary_sha256": file_hash(binary), "reasoning_capable": True,
              "runtime_config": {"backend": "cpu", "threads": 1, "parallel": 1, "special": False, "backend_sampling": False}}
    def factory(broken):
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"build_info": "a" * 40, "chat_template": "mock", "total_slots": 1})
            body = json.loads(request.content)
            assert body["return_tokens"] is True
            text = json.loads(body["grammar"].removeprefix("root ::= "))
            tokens = list(text.encode())[:body.get("n_predict", body.get("max_tokens"))]
            payload = native(tokens, stop=True, content=text)
            if broken and body["stream"] and 226 in tokens:
                payload["completion_probabilities"].pop(tokens.index(226))
            if request.url.path == "/v1/chat/completions":
                choice = {"finish_reason": "stop", "logprobs": {"content": [record(i) for i in tokens]}}
                if chat_usage and broken and 226 in tokens:
                    choice["logprobs"]["content"].pop(tokens.index(226))
                choice["delta" if body["stream"] else "message"] = {"content": text}
                payload = {"choices": [choice]}
                if chat_usage and not body['stream']:
                    payload['usage'] = {'completion_tokens': len(tokens)}
            content = sse([payload]) if body["stream"] else json.dumps(payload).encode()
            if body["stream"] and request.url.path == "/v1/chat/completions":
                if chat_usage:
                    content += sse([{'choices': [], 'usage': {'completion_tokens': len(tokens)}}])
                content += b"data: [DONE]\n\n"
            return httpx.Response(200, content=content)
        return lambda url: Client(url, transport=httpx.MockTransport(handler))
    good = run(build_dir, target, client_factory=factory(False))
    bad = run(build_dir, {**target, "target_id": "mock-unpatched"}, client_factory=factory(True))
    monkeypatch.setattr("murphy.core.compare.Vocabulary", lambda path: ByteVocabulary())
    return build_dir, good, bad


def test_end_to_end_positive_negative_and_identity_gates(pipeline):
    build_dir, good, bad = pipeline
    assert good != bad
    assert good == build_dir / 'runs' / 'mock-patched'
    assert bad == build_dir / 'runs' / 'mock-unpatched'
    assert len(read_jsonl(good / "responses.jsonl", "response")) == 64
    grade(build_dir, good)
    with pytest.raises(FileExistsError):
        grade(build_dir, good)
    output = compare(build_dir, bad, good)
    assert output.name == 'mock-unpatched__mock-patched.jsonl'
    comparisons = read_jsonl(output, "comparison")
    assert len(comparisons) == 32
    assert all(r["schema_version"] == "2" for r in comparisons)
    names = {"token_level_coverage", "complete_response_rate", "numerical_agreement", "actual_output_exposure", "task_compliance"}
    assert all(set(r["measurements"]) == names for r in comparisons)
    ascii_row = next(r for r in comparisons if r["task_id"] == "ascii" and r["mode"] == "stream")
    assert ascii_row["measurements"]["actual_output_exposure"]["right"]["value"] == 1
    old = deepcopy(ascii_row)
    old["schema_version"] = "1"
    old.pop("measurements")
    old.pop("references")
    validate(old, "comparison")  # Existing v1 artifacts retain their reader.
    invalid = deepcopy(ascii_row)
    invalid["measurements"]["numerical_agreement"]["value"] = None
    with pytest.raises(ValidationError):
        validate(invalid, "comparison")
    invalid = deepcopy(ascii_row)
    invalid["measurements"]["made_up_metric"] = {}
    with pytest.raises(ValidationError):
        validate(invalid, "comparison")
    assert any(c["status"] == "fail" for r in comparisons for repeat in r["left"]["checks_by_repeat"] for c in repeat)
    assert not any(c["status"] == "fail" for r in comparisons for repeat in r["right"]["checks_by_repeat"] for c in repeat)
    with pytest.raises(FileExistsError):
        compare(build_dir, bad, good)
    # Reject edited derived observations even if their JSON schema is valid.
    path = good / "responses.jsonl"
    original = path.read_text()
    changed = original.splitlines()
    first = json.loads(changed[0])
    first["chunks"][0]["content"] = "edited"
    changed[0] = json.dumps(first)
    path.write_text("\n".join(changed) + "\n")
    with pytest.raises(ValueError, match="preserved response bytes"):
        load_run(build_dir, good)
    path.write_text(original)
    # A missing attempt is not a successful partial benchmark.
    path = good / "responses.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:-1]))
    with pytest.raises(ValueError, match="attempt rows"):
        load_run(build_dir, good)


def test_repeat_run_rejected_before_contacting_server(pipeline):
    build_dir, good, _ = pipeline
    target = json.loads((good / 'target.json').read_text())
    before = (good / 'responses.jsonl').read_bytes()
    def forbidden(_):
        pytest.fail('Existing run must fail before network access')
    with pytest.raises(FileExistsError):
        run(build_dir, target, client_factory=forbidden)
    assert (good / 'responses.jsonl').read_bytes() == before


def test_historical_nested_runs_remain_readable(pipeline):
    build_dir, good, bad = pipeline
    moved = []
    for path in (good, bad):
        meta = json.loads((path / 'run.json').read_text())
        temporary = path.with_name(path.name + '-moving')
        path.rename(temporary)
        path.mkdir()
        destination = path / meta['run_id']
        temporary.rename(destination)
        load_run(build_dir, destination)
        moved.append(destination)
    output = compare(build_dir, moved[1], moved[0])
    assert output.name == moved[1].name + '__' + moved[0].name + '.jsonl'


def test_comparison_name_independent_of_relative_paths(pipeline, monkeypatch):
    build_dir, good, bad = pipeline
    monkeypatch.chdir(build_dir.parent)
    output = compare(build_dir.name, bad, good)
    assert output.name == 'mock-unpatched__mock-patched.jsonl'


@pytest.mark.parametrize("field,value", [("target_id", "different"), ("model_sha256", "0" * 64),
                                         ("binary_sha256", "0" * 64), ("source_commit", "b" * 40)])
def test_saved_target_is_bound_to_run(pipeline, field, value):
    build_dir, good, _ = pipeline
    path = good / "target.json"
    target = json.loads(path.read_text())
    target[field] = value
    path.write_text(json.dumps(target))
    with pytest.raises(ValueError, match="target disagrees"):
        load_run(build_dir, good)


@pytest.mark.parametrize("changes", [
    {"http_status": 500}, {"http_status": None}, {"error": "silently ignored"},
    {"status": "http_error", "error": "HTTP failure"}, {"status": "timeout", "error": None},
])
def test_saved_transport_status_cannot_be_relabelled(pipeline, changes):
    build_dir, good, _ = pipeline
    path = good / "responses.jsonl"
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first.update(changes)
    lines[0] = json.dumps(first)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        load_run(build_dir, good)


@pytest.mark.parametrize("info,slots", [("a" * 7 + "b" * 33, 1), ("b" * 40, 1),
                                       ("a" * 40, 2), ("a" * 40, None), ("a" * 40, True)])
def test_observed_revision_and_slot_gates(info, slots):
    with pytest.raises(ValueError):
        check_properties({"source_commit": "a" * 40}, {"build_info": info, "total_slots": slots})


def test_abbreviated_observed_revision_is_allowed_but_not_attestation():
    check_properties({"source_commit": "a" * 40}, {"build_info": "b1234 (aaaaaaaa)", "total_slots": 1})


@pytest.mark.parametrize("changes", [{"value": 0.5}, {"numerator": 999},
                                    {"status": "unavailable", "value": None}])
def test_metric_fraction_arithmetic_is_validated(pipeline, changes):
    build_dir, good, bad = pipeline
    row = read_jsonl(compare(build_dir, bad, good), "comparison")[0]
    row["measurements"]["complete_response_rate"]["right"].update(changes)
    with pytest.raises(ValueError, match="measurement fraction"):
        validate(row, "comparison")
