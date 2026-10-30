"""Independent GGUF byte-piece oracle, with explicit tokenizer admission."""
import re

from gguf import GGUFReader

from murphy.util.io import digest


def bytelevel_decoder():
    """Inverse of GPT-2's reversible byte alphabet (not UTF-8 token labels)."""
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = values.copy()
    next_code = 256
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            alphabet.append(next_code)
            next_code += 1
    return {chr(code): byte for code, byte in zip(alphabet, values)}


BYTELEVEL = bytelevel_decoder()


def bpe_bytes(piece):
    try:
        return [BYTELEVEL[char] for char in piece]
    except KeyError as exc:
        raise ValueError("Non-bytelevel symbol in normal BPE piece") from exc


class Vocabulary:
    def __init__(self, path):
        reader = GGUFReader(str(path))
        def field(name):
            item = reader.get_field(name)
            if item is None:
                raise ValueError(f"GGUF lacks required vocabulary field: {name}")
            return item.contents()
        model = field("tokenizer.ggml.model")
        if model not in ("llama", "gpt2"):
            raise ValueError(f"Unqualified tokenizer byte decoder: {model}")
        if model == "gpt2":
            pre = field("tokenizer.ggml.pre")
            if pre not in ("qwen2", "qwen35"):
                raise ValueError("Unqualified tokenizer BPE pre-type")
            escapes = reader.get_field("tokenizer.ggml.escape_whitespaces")
            if escapes is not None and escapes.contents():
                raise ValueError("Unqualified tokenizer whitespace escaping")
        self.pieces = field("tokenizer.ggml.tokens")
        self.types = field("tokenizer.ggml.token_type")
        if not isinstance(self.pieces, list) or not isinstance(self.types, list) or not self.pieces or len(self.pieces) != len(self.types):
            raise ValueError("Empty vocabulary or token/type length mismatch")
        if any(not isinstance(p, str) for p in self.pieces) or any(type(t) is not int or t not in range(1, 7) for t in self.types):
            raise ValueError("Malformed vocabulary pieces/types")
        self.model = model
        decoder = pre + "-bytelevel-special-false-v1" if model == "gpt2" else "sentencepiece-special-false-v1"
        self.identity = {"decoder": decoder, "tokenizer_model": model,
                         "token_count": len(self.pieces),
                         "vocabulary_sha256": digest({"model": model, "pieces": self.pieces, "types": self.types})}

    def token_bytes(self, token_id):
        if type(token_id) is not int or not 0 <= token_id < len(self.pieces):
            raise ValueError("Invalid vocabulary ID")
        piece, kind = self.pieces[token_id], self.types[token_id]
        if kind in (2, 3):  # UNKNOWN, CONTROL with special=false.
            return []
        if kind == 6:
            if not re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece):
                raise ValueError("Malformed byte-fallback piece")
            return [int(piece[3:5], 16)]
        if kind == 1:
            if getattr(self, "model", "llama") == "gpt2":
                return bpe_bytes(piece)
            return list(piece.replace("▁", " ").encode("utf-8"))
        if kind == 4:
            return list(piece.encode("utf-8"))
        if kind == 5:  # UNUSED is suppressed by the pinned implementation.
            return []
        raise ValueError("Unqualified token type")
