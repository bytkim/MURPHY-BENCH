import base64
import json
from pathlib import Path

import httpx
import pytest

from murphy.client import Client, joined, parse_body
from conftest import native, sse


def test_template_control_stays_in_client_and_rejects_bad_shape():
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"prompt": "wrapped €"})
    with Client("http://127.0.0.1:1", transport=httpx.MockTransport(handler)) as client:
        assert client.apply_template("€") == "wrapped €"
    assert seen == [{"messages": [{"role": "user", "content": "€"}], "chat_template_kwargs": {"enable_thinking": False}}]
    with Client("http://127.0.0.1:1", transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"prompt": []}))) as client:
        with pytest.raises(ValueError, match="Malformed"):
            client.apply_template("€")


def test_chunk_boundaries_and_terminal(observed):
    assert observed["status"] == "complete"
    assert len(observed["chunks"]) == 2
    assert joined(observed)["token_ids"] == [65, 226, 130, 172]
    assert observed["chunks"][1]["token_ids"] == [226, 130, 172]


def test_incomplete_and_malformed_preserve_prior_events():
    prefix = sse([native([65], content="A")])
    assert parse_body(prefix, "/completion", True)["status"] == "incomplete"
    result = parse_body(prefix + b"data: {bad}\n\n", "/completion", True)
    assert result["status"] == "parse_error"
    assert len(result["events"]) == 2
    assert joined(result)["token_ids"] == [65]
    assert parse_body(prefix + b"data: {}", "/completion", True)["status"] == "parse_error"


def test_checked_in_incomplete_fixture():
    body = (Path(__file__).parent / "fixtures/incomplete.sse").read_bytes()
    result = parse_body(body, "/completion", True)
    assert result["status"] == "incomplete"
    assert joined(result)["token_ids"] == [65]


def test_chat_reasoning_empty_delta_usage_and_done():
    payloads = [
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"delta": {"reasoning_content": "thinking"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "logprobs": {"content": [{"id": 226, "bytes": [226]}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "€"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 3}},
    ]
    body = sse(payloads) + b"data: [DONE]\n\n"
    result = parse_body(body, "/v1/chat/completions", True)
    assert result["status"] == "complete"
    assert len(result["events"]) == 6
    assert joined(result)["token_ids"] is None
    assert joined(result)["reasoning_content"] == "thinking"
    assert joined(result)["records"][0]["bytes"] == [226]
    assert parse_body(sse(payloads), "/v1/chat/completions", True)["status"] == "incomplete"


def test_multiline_crlf_comments_and_split_wire_utf8():
    body = b': keepalive\r\n\r\ndata: {"content":"\xe2\x82\xac",\r\ndata: "tokens":[],"stop":true}\r\n\r\n'
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for byte in body:
                yield bytes([byte])
    with Client("http://127.0.0.1:1", transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Stream()))) as client:
        result = client.complete("/completion", {"stream": True, "return_tokens": True})
    assert result["status"] == "complete"
    assert base64.b64decode(result["body_base64"]) == body
    assert result["events"][0]["event"] == "comment"


def test_timeout_keeps_received_bytes_no_retry():
    body = sse([native([65])])
    calls = []
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield body
            raise httpx.ReadTimeout("timeout")
    def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=Stream())
    with Client("http://127.0.0.1:1", transport=httpx.MockTransport(handler)) as client:
        result = client.complete("/completion", {"stream": True, "return_tokens": True})
    assert result["status"] == "timeout"
    assert base64.b64decode(result["body_base64"]) == body
    assert joined(result)["token_ids"] == [65]
    assert len(calls) == 1


@pytest.mark.parametrize("body", [b'{"stop":true,"x":NaN}', b'{"stop":true,"x":1e999}',
                                 b'{"stop":true,"content":"\\ud800"}', b'\xff'])
def test_bad_json_is_observation(body):
    with Client("http://127.0.0.1:1", transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body))) as client:
        result = client.complete("/completion", {"return_tokens": True})
    assert result["status"] == "parse_error"
    assert base64.b64decode(result["body_base64"]) == body


def test_http_error_and_required_tokens():
    with Client("http://127.0.0.1:1", transport=httpx.MockTransport(lambda _: httpx.Response(500, content=b"failed"))) as client:
        result = client.complete("/completion", {"return_tokens": True})
        assert result["status"] == "http_error"
        with pytest.raises(ValueError):
            client.complete("/completion", {})


def test_partial_utf8_tail_keeps_complete_prior_chunks():
    body = sse([native([65])]) + b'data: {"content":"\xe2'
    result = parse_body(body, "/completion", True)
    assert result["status"] == "parse_error"
    assert joined(result)["token_ids"] == [65]


@pytest.mark.parametrize("payload", [{"tokens": 1, "stop": True}, {"content": ["bad"], "stop": True}])
def test_bad_envelope_is_captured_not_repaired(payload):
    result = parse_body(json.dumps(payload).encode(), "/completion", False)
    assert result["status"] == "parse_error"
    assert result["events"][0]["payload"] == payload


@pytest.mark.parametrize("field", ["content", "reasoning_content"])
@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_falsey_chat_text_is_not_normalized(field, value):
    payload = {"choices": [{"message": {field: value}, "finish_reason": "stop"}]}
    result = parse_body(json.dumps(payload).encode(), "/v1/chat/completions", False)
    assert result["status"] == "parse_error"
    assert result["events"][0]["payload"] == payload
    assert result["chunks"] == []


@pytest.mark.parametrize("value", [False, 0, [], ""])
def test_malformed_chat_logprobs_are_not_empty(value):
    payload = {"choices": [{"message": {"content": "x"}, "logprobs": value, "finish_reason": "stop"}]}
    assert parse_body(json.dumps(payload).encode(), "/v1/chat/completions", False)["status"] == "parse_error"


def test_null_chat_fields_and_empty_deltas_remain_legal():
    payload = {"choices": [{"delta": {"content": None, "reasoning_content": None},
                           "logprobs": None, "finish_reason": "stop"}]}
    result = parse_body(sse([payload]) + b"data: [DONE]\n\n", "/v1/chat/completions", True)
    assert result["status"] == "complete"
    assert joined(result)["content"] == joined(result)["reasoning_content"] == ""


@pytest.mark.parametrize("endpoint,payload", [
    ("/completion", {"stop": 1}),
    ("/completion", {"stop": True, "stop_type": ""}),
    ("/v1/chat/completions", {"choices": [{"message": {}, "finish_reason": ""}]}),
])
def test_malformed_terminal_markers_are_not_completion(endpoint, payload):
    assert parse_body(json.dumps(payload).encode(), endpoint, False)["status"] == "parse_error"


@pytest.mark.parametrize("endpoint", ["/completion", "/v1/chat/completions"])
def test_generation_after_terminal_is_rejected_with_raw_event_retained(endpoint):
    payload = native([65], stop=True, content="A") if endpoint == "/completion" else {
        "choices": [{"delta": {"content": "A"}, "finish_reason": "stop"}]}
    result = parse_body(sse([payload, payload]), endpoint, True)
    assert result["status"] == "parse_error"
    assert len(result["events"]) == 2 and len(result["chunks"]) == 1
    assert result["events"][1]["payload"] == payload


def test_duplicate_done_is_rejected():
    payload = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    result = parse_body(sse([payload]) + b"data: [DONE]\n\ndata: [DONE]\n\n", "/v1/chat/completions", True)
    assert result["status"] == "parse_error"
    assert len(result["events"]) == 3
