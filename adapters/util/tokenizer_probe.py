"""In-context byte boundary measurements; no per-character retokenization."""
import bisect
import unicodedata
from pathlib import Path

from jsonschema import Draft202012Validator

from murphy.util.io import encoded, strict_loads
from murphy.util.vocabulary import bpe_bytes


def write_probe(path, records):
    """Validate adapter-owned selection evidence, outside the core contracts."""
    schema = Path(__file__).resolve().parents[1] / 'contract/tokenizer-probe/tokenizer-probe-v1.schema.json'
    value = {'schema_version': '1', 'records': records}
    payload = encoded(value)
    Draft202012Validator(strict_loads(schema.read_text())).validate(value)
    with Path(path).open('xb') as stream:
        stream.write(payload)


def split_observation(text, token_ids, token_bytes):
    pieces = [bytes(token_bytes(i)) for i in token_ids]
    if b"".join(pieces) != text.encode("utf-8"):
        raise ValueError("Token bytes do not reconstruct measured text")
    ends, offset = [], 0
    for char in text:
        offset += len(char.encode("utf-8"))
        ends.append(offset)
    boundaries, offset = set(), 0
    for piece in pieces[:-1]:
        offset += len(piece)
        if offset not in ends and 0 < offset < len(text.encode("utf-8")):
            boundaries.add(bisect.bisect_right(ends, offset))
    return {"characters": len(text), "split_characters": len(boundaries), "tokens": len(token_ids)}


def measure(texts, tokenizer, *, sequences=False):
    result = {"texts": 0, "characters": 0, "split_characters": 0, "tokens": 0, "nfc_changed_texts": 0}
    sequence_splits = 0
    for text in texts:
        # The selected HF tokenizer normalizes NFC. Count this explicitly; tasks remain unmodified.
        normalized = unicodedata.normalize("NFC", text)
        result["nfc_changed_texts"] += normalized != text
        encoding = tokenizer.encode(text, add_special_tokens=False)
        observation = split_observation(normalized, encoding.ids, lambda i: bpe_bytes(tokenizer.id_to_token(i)))
        result["texts"] += 1
        for key in ("characters", "split_characters", "tokens"):
            result[key] += observation[key]
        sequence_splits += len(encoding.ids) > 1
    result["split_character_rate"] = result["split_characters"] / result["characters"] if result["characters"] else None
    if sequences:
        result.update(split_sequences=sequence_splits, split_sequence_rate=sequence_splits / result["texts"])
    return result
