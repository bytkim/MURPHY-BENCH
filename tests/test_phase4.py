import pytest
from adapters.util.prompt_templates import wrapper_for

from adapters.core.common import aligned_ids, index_rows
from adapters.datasets.unicode_emoji import family
from adapters.datasets.belebele import rendered
from murphy.core.run import request_for
from adapters.util.tokenizer_probe import split_observation


def test_context_boundary_counts_characters_not_cuts():
    data = {0: b"A\xe2", 1: b"\x82", 2: b"\xacB"}
    assert split_observation("A€B", [0, 1, 2], data.__getitem__) == {"characters": 3, "split_characters": 1, "tokens": 3}
    assert split_observation("AB", [0], lambda _: b"AB")["split_characters"] == 0
    with pytest.raises(ValueError, match="reconstruct"):
        split_observation("€", [0], lambda _: b"?")


@pytest.mark.parametrize("sequence,expected", [("🇯🇵", "flags-including-tags"),
    ("🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f", "flags-including-tags"),
    ("👩🏽\u200d💻", "zwj"), ("👍🏽", "modifier-without-zwj"), ("🙂", "single-code-point"),
    ("1️⃣", "other-multi-code-point"), ("☀️", "other-multi-code-point")])
def test_emoji_family_precedence(sequence, expected):
    assert family(sequence) == expected


def test_alignment_rejects_missing_and_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        index_rows([{"id": 1}, {"id": 1}], lambda r: r["id"])
    with pytest.raises(ValueError, match="differ"):
        aligned_ids({"a": {1: {}}, "b": {2: {}}}, 1, "x")
    tables = {"a": {i: {} for i in range(20)}, "b": {i: {} for i in reversed(range(20))}}
    assert aligned_ids(tables, 3, "x") == aligned_ids(dict(reversed(list(tables.items()))), 3, "x")


def test_original_choice_language_and_answer_key():
    row = {"flores_passage": "文章", "question": "質問", "correct_answer_num": "2",
           **{f"mc_answer{i}": f"答え{i}" for i in range(1, 5)}}
    assert rendered(row)[1] == "B. 答え2"
    row["correct_answer_num"] = "0"
    with pytest.raises(ValueError):
        rendered(row)


def test_wrapper_is_recorded_request_only():
    wrapper = wrapper_for("U{{ messages[0].content }}A", ["€", "中文"])
    condition = {"endpoint": "/completion", "stop": [], "literal_grammar": False,
                 "max_tokens": 20, "prompt_wrapper": wrapper}
    task = {"prompt": "€"}
    request = request_for(task, condition, "stream")
    assert request["prompt"] == "U€A" and request["return_tokens"] is True
    assert task == {"prompt": "€"} and "grammar" not in request
    with pytest.raises(ValueError, match="differs"):
        wrapper_for("{{ messages[0].content | trim }}", [" whitespace "])




def test_source_admission_hash_checks_before_parse(tmp_path):
    import json
    from adapters.core.common import Sources
    from murphy.util.io import file_hash
    (tmp_path/"preservation").mkdir()
    path = tmp_path/"sample.jsonl"
    path.write_text('{"id": 1}\n')
    (tmp_path/"preservation/source-manifest.json").write_text(json.dumps({"files": [
        {"path": "sample.jsonl", "sha256": file_hash(path), "bytes": path.stat().st_size}]}))
    source = Sources(tmp_path)
    assert source.rows("sample.jsonl") == [{"id": 1}]
    path.write_text('{"id": 2}\n')
    with pytest.raises(ValueError, match="identity mismatch"):
        source.rows("sample.jsonl")
