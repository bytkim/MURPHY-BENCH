import os
from pathlib import Path
os.environ["MURPHY_SOURCE"] = str(Path(__file__).resolve().parents[1])

import json

import pytest

from murphy.client import parse_body


class ByteVocabulary:
    identity = {"decoder": "test-byte-v1", "tokenizer_model": "test-only", "token_count": 256,
                "vocabulary_sha256": "0" * 64}
    def token_bytes(self, token_id):
        if type(token_id) is not int or not 0 <= token_id < 256:
            raise ValueError("Invalid vocabulary ID")
        return [token_id]


def record(token, value=-0.1):
    return {"id": token, "bytes": [token], "token": "", "logprob": value,
            "top_logprobs": [{"id": token, "bytes": [token], "token": "", "logprob": value}]}


def native(tokens, *, stop=False, content=""):
    return {"tokens": tokens, "completion_probabilities": [record(i) for i in tokens],
            "content": content, "stop": stop, "stop_type": "eos" if stop else None}


def sse(payloads):
    return b"".join(b"data: " + json.dumps(p, ensure_ascii=False).encode() + b"\n\n" for p in payloads)


@pytest.fixture
def observed():
    return parse_body(sse([native([65], content="A"), native([226, 130, 172], stop=True, content="€")]), "/completion", True)
