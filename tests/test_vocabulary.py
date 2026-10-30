import pytest
from gguf import GGUFWriter

from murphy.util.vocabulary import Vocabulary
from murphy.util.io import file_hash


def write_vocabulary(path, *, model="llama", pieces=None, types=None, include_types=True, pre=None):
    pieces = pieces or ["<unk>", "<s>", "</s>", *[f"<0x{i:02X}>" for i in range(256)], "▁A", "€", "👩\u200d💻"]
    types = types or [2, 3, 3, *[6] * 256, 1, 1, 4]
    writer = GGUFWriter(str(path), "llama")
    writer.add_tokenizer_model(model)
    if pre:
        writer.add_tokenizer_pre(pre)
    writer.add_token_list(pieces)
    if include_types:
        writer.add_token_types(types)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_reads_actual_gguf_and_all_byte_values(tmp_path):
    path = tmp_path / "vocabulary.gguf"
    write_vocabulary(path)
    before = file_hash(path)
    vocab = Vocabulary(path)
    assert [vocab.token_bytes(i + 3) for i in range(256)] == [[i] for i in range(256)]
    assert vocab.token_bytes(0) == vocab.token_bytes(1) == vocab.token_bytes(2) == []
    assert vocab.token_bytes(259) == [32, 65]
    assert vocab.token_bytes(260) == [226, 130, 172]
    assert vocab.token_bytes(261) == list("👩\u200d💻".encode())
    assert vocab.identity["token_count"] == 262
    assert len(vocab.identity["vocabulary_sha256"]) == 64
    assert file_hash(path) == before
    assert Vocabulary(path).identity == vocab.identity


@pytest.mark.parametrize("options,reason", [
    ({"model": "gpt2", "pre": "other"}, "Unqualified tokenizer"),
    ({"include_types": False}, "lacks required"),
    ({"pieces": ["a", "b"], "types": [1]}, "length mismatch"),
    ({"pieces": ["a"], "types": [99]}, "Malformed vocabulary"),
])
def test_unqualified_or_incomplete_gguf_fails_closed(tmp_path, options, reason):
    path = tmp_path / "bad.gguf"
    write_vocabulary(path, **options)
    with pytest.raises(ValueError, match=reason):
        Vocabulary(path)


def test_vocabulary_content_changes_identity(tmp_path):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    write_vocabulary(a, pieces=["A"], types=[1])
    write_vocabulary(b, pieces=["B"], types=[1])
    assert Vocabulary(a).identity["vocabulary_sha256"] != Vocabulary(b).identity["vocabulary_sha256"]


def test_sentencepiece_decoding_not_naive_utf8():
    vocab = Vocabulary.__new__(Vocabulary)
    vocab.pieces = ["▁A", "<0xE2>", "</s>", "<user>", "<unused>"]
    vocab.types = [1, 6, 3, 4, 5]
    assert vocab.token_bytes(0) == [32, 65]
    assert vocab.token_bytes(1) == [226]
    assert vocab.token_bytes(2) == []
    assert vocab.token_bytes(3) == list(b"<user>")
    assert vocab.token_bytes(4) == []
    with pytest.raises(ValueError):
        vocab.token_bytes(True)
    with pytest.raises(ValueError):
        vocab.token_bytes(-1)


@pytest.mark.parametrize("pre", ["qwen2", "qwen35"])
def test_qwen_bytelevel_gguf_all_bytes_and_special_policy(tmp_path, pre):
    from murphy.util.vocabulary import BYTELEVEL
    inverse = {byte: piece for piece, byte in BYTELEVEL.items()}
    path = tmp_path / "qwen.gguf"
    write_vocabulary(path, model="gpt2", pre=pre,
                     pieces=[inverse[b] for b in range(256)] + ["<special>", "<user>", "[PAD]"],
                     types=[1] * 256 + [3, 4, 5])
    vocab = Vocabulary(path)
    assert [vocab.token_bytes(i) for i in range(256)] == [[i] for i in range(256)]
    assert inverse[32] == "Ġ" and inverse[10] == "Ċ" and inverse[0] == "Ā"
    assert vocab.token_bytes(256) == vocab.token_bytes(258) == []
    assert vocab.token_bytes(257) == list(b"<user>")
    assert vocab.identity["decoder"] == pre + "-bytelevel-special-false-v1"
