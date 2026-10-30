"""Only HTTP boundary: capture entity bytes and SSE events, never repair records."""
import base64
import time
from datetime import datetime, timezone

import httpx

from murphy.util.io import strict_loads


def now():
    return datetime.now(timezone.utc).isoformat()


def chunk_view(payload, endpoint):
    if not isinstance(payload, dict) or "error" in payload:
        raise ValueError("Expected a completion object, not an error")
    if endpoint == "/completion":
        if "stop" in payload and type(payload["stop"]) is not bool:
            raise ValueError("Malformed native stop flag")
        result = {"token_ids": payload.get("tokens"), "records": payload.get("completion_probabilities"),
                "content": payload.get("content", ""), "reasoning_content": "",
                "finish": payload.get("stop_type", "stop") if payload.get("stop") is True else None}
    elif endpoint == "/v1/chat/completions":
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("Expected zero or one chat choice")
        choice = choices[0] if choices else {}
        if not isinstance(choice, dict):
            raise ValueError("Malformed chat choice")
        message = choice.get("delta", choice.get("message", {}))
        logprobs = choice.get("logprobs")
        if not isinstance(message, dict) or (logprobs is not None and not isinstance(logprobs, dict)):
            raise ValueError("Malformed chat message or logprobs")
        result = {"token_ids": None, "records": logprobs.get("content") if logprobs is not None else None,
                  "content": message.get("content"), "reasoning_content": message.get("reasoning_content"),
                  "finish": choice.get("finish_reason")}
        # Missing/null text is legal for role, reasoning and usage-only deltas.
        # Other false-y values are malformed evidence, not empty strings.
        for field in ("content", "reasoning_content"):
            if result[field] is None:
                result[field] = ""
    else:
        raise ValueError("Unsupported completion endpoint")
    for field in ("token_ids", "records"):
        if result[field] is not None and not isinstance(result[field], list):
            raise ValueError(f"Malformed {field}")
    if any(not isinstance(result[field], str) for field in ("content", "reasoning_content")):
        raise ValueError("Malformed content")
    if result["finish"] is not None and (not isinstance(result["finish"], str) or not result["finish"]):
        raise ValueError("Malformed finish reason")
    return result


def parse_body(body, endpoint, streaming):
    events, chunks = [], []
    done = False
    error = None
    try:
        if not streaming:
            text = body.decode("utf-8")
            payload = strict_loads(text)
            events.append({"data": text, "payload": payload, "event": "json"})
            chunks.append(chunk_view(payload, endpoint))
        else:
            # Split only SSE line endings, not Unicode text separators.
            text = body.removeprefix(b"\xef\xbb\xbf").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            data, event, lines = [], "message", []
            for raw_line in text.split(b"\n")[:-1]:
                line = raw_line.decode("utf-8")
                if line:
                    lines.append(line)
                    field, _, value = line.partition(":")
                    value = value.removeprefix(" ")
                    if field == "data":
                        data.append(value)
                    elif field == "event":
                        event = value
                    continue
                joined = "\n".join(data)
                observed = {"data": joined, "payload": None, "event": event if data else "comment"}
                events.append(observed)
                if joined == "[DONE]":
                    if done:
                        raise ValueError("Duplicate SSE DONE")
                    done = True
                elif data:
                    if done:
                        raise ValueError("Data after SSE DONE")
                    payload = strict_loads(joined)
                    observed["payload"] = payload
                    chunk = chunk_view(payload, endpoint)
                    if any(c["finish"] is not None for c in chunks) and (endpoint == "/completion" or payload["choices"]):
                        raise ValueError("Completion data after terminal chunk")
                    chunks.append(chunk)
                data, event, lines = [], "message", []
            if lines or text.split(b"\n")[-1]:
                error = "Unterminated SSE event"
    except (ValueError, TypeError, AttributeError) as exc:
        error = str(exc)
    terminal = any(c["finish"] is not None for c in chunks)
    complete = terminal and (not streaming or endpoint == "/completion" or done)
    status = "parse_error" if error else ("complete" if complete else "incomplete")
    return {"events": events, "chunks": chunks, "status": status, "error": error}


def joined(response):
    chunks = response["chunks"]
    ids_available = bool(chunks) and all(c["token_ids"] is not None for c in chunks)
    return {"content": "".join(c["content"] for c in chunks),
            "reasoning_content": "".join(c["reasoning_content"] for c in chunks),
            "token_ids": [i for c in chunks for i in c["token_ids"]] if ids_available else None,
            "records": [r for c in chunks for r in (c["records"] or [])]}


def completion_accounting(response):
    """Whole-response chat usage, derived from captured events without repair.

    This is server accounting, not a token identity or per-chunk reference.
    Null usage on intermediate deltas is legal; multiple non-null observations
    are ambiguous even if equal. Partial transports never supply a denominator.
    """
    result = {"count": None, "reference": "chat_usage_completion_tokens",
              "scope": "whole_response", "event_index": None, "reason": None}
    if response.get("condition", {}).get("endpoint") != "/v1/chat/completions":
        return {**result, "reason": "not_chat"}
    if response["status"] != "complete":
        return {**result, "reason": "incomplete_transport"}
    usage = [(i, e["payload"]) for i, e in enumerate(response["events"])
             if isinstance(e["payload"], dict) and e["payload"].get("usage") is not None]
    if len(usage) != 1:
        return {**result, "reason": "usage_missing" if not usage else "usage_ambiguous"}
    index, payload = usage[0]
    result["event_index"] = index
    value = payload["usage"]
    count = value.get("completion_tokens") if isinstance(value, dict) else None
    if type(count) is not int or count < 0:
        return {**result, "reason": "usage_malformed"}
    if response["mode"] == "stream":
        # The final usage-only event follows the terminal choice, before DONE.
        previous = [e["payload"] for e in response["events"][:index] if isinstance(e["payload"], dict)]
        terminal = any(c.get("finish_reason") for p in previous for c in p.get("choices", []))
        later = any(isinstance(e["payload"], dict) for e in response["events"][index + 1:])
        if payload.get("choices") != [] or not terminal or later:
            return {**result, "reason": "usage_not_final"}
    return {**result, "count": count}


class Client:
    def __init__(self, base_url, *, transport=None, timeout=30, max_bytes=8 * 1024 * 1024):
        self.http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport,
                                 follow_redirects=False, trust_env=False,
                                 headers={"Accept-Encoding": "identity"})
        self.max_bytes = max_bytes

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.http.close()

    def properties(self):
        response = self.http.get("/props")
        response.raise_for_status()
        props = strict_loads(response.content)
        # Never save optional connection codes or authentication material.
        keys = ("build_info", "chat_template", "chat_template_caps", "total_slots", "model_alias")
        if not isinstance(props.get("build_info"), str):
            raise ValueError("Server did not expose build_info")
        return {k: props[k] for k in keys if k in props}

    def apply_template(self, prompt, *, thinking=False):
        response = self.http.post("/apply-template", json={"messages": [{"role": "user", "content": prompt}],
                                  "chat_template_kwargs": {"enable_thinking": thinking}})
        response.raise_for_status()
        value = strict_loads(response.content)
        if not isinstance(value, dict) or not isinstance(value.get("prompt"), str):
            raise ValueError("Malformed rendered template")
        return value["prompt"]

    def complete(self, endpoint, request):
        if request.get("return_tokens") is not True:
            raise ValueError("return_tokens=true is required")
        started_at, start = now(), time.monotonic()
        body, status_code, failure, error = bytearray(), None, None, None
        try:
            with self.http.stream("POST", endpoint, json=request) as response:
                status_code = response.status_code
                for piece in response.iter_bytes():
                    remaining = self.max_bytes - len(body)
                    body.extend(piece[:remaining])
                    if len(piece) > remaining:
                        failure, error = "body_limit", "Response exceeded capture limit"
                        break
                    if time.monotonic() - start > 60:
                        failure, error = "timeout", "60-second capture deadline; no retry"
                        break
        except httpx.TimeoutException:
            failure, error = "timeout", "HTTP timeout; no retry"
        except httpx.HTTPError as exc:
            failure, error = "connection_error", type(exc).__name__ + "; no retry"
        parsed = parse_body(bytes(body), endpoint, request.get("stream", False))
        if status_code is not None and status_code != 200:
            failure, error = "http_error", f"HTTP {status_code}; no retry"
        if failure:
            parsed.update(status=failure, error=error)
        return {"started_at": started_at, "elapsed_seconds": time.monotonic() - start,
                "http_status": status_code, "body_base64": base64.b64encode(body).decode(), **parsed}
