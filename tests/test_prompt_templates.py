import copy
from adapters.core.prepare import parse_prompt_revisions

import pytest
from jsonschema import ValidationError

from adapters.util.fixture import build
from adapters.util.prompt_templates import PromptTemplate, load
from adapters.datasets.unicode_emoji import tasks as emoji_tasks
from murphy.util.io import digest, load_build, validate


@pytest.mark.parametrize("name,revision,expected", [
    ("belebele", 2, "ec95e5a379e32ebb8e0cba345466c8e60838014c8e372733f6bc2d4b6a497b8e"),
    ("flores_plus", 1, "ed88a6b44744fc1edaeb07452e5a8814ec34abf5ac6ea3228faa3101c8ea1fb4"),
    ("belebele", 1, "afe05d92b72bb246b4e10afc62d25f034487eb56f252e0e8dd91e9376b52cf86"),
    ("unicode_emoji", 1, "fd35f8757812c016b5e8c5588cbed9484f0a4be3ceeb3f06a5194a2c8ff7ffcd"),
    ("unicode_emoji", 2, "6296c603c092225384c9354f440666c53f27666fe9599ec09a42f056f5efac08"),
    ("unicode_emoji", 3, "4998268c27ccae6316fb755550ee4314a6db71ca332e04971eca33ea36f2d0ec"),
])
def test_published_revisions_are_append_only(name, revision, expected):
    # Do not update these identities to edit old wording: add a new revision.
    assert load(name, revision).identity == {"revision": revision, "sha256": expected}


def test_v1_is_byte_exact_and_does_not_expand_source_text():
    text = "  日本語 $name {braces}\n"
    assert load("flores_plus").render(text=text) == "Repeat the following sentence exactly. Return only the sentence.\n" + text
    assert load("belebele").render(text=text) == (
        "Read the passage and choose the correct answer. Return only the answer letter, a period, a space, "
        "and the complete original text of that choice, preserving its language.\n\n" + text)
    assert load("unicode_emoji").render(sequence="🙂", name="slightly smiling face") == (
        "Output exactly the emoji followed by its supplied English name, using {emoji | name}. "
        "Return only that single entry.\nEmoji: 🙂\nEnglish name: slightly smiling face")


@pytest.mark.parametrize("name,revision", [("../escape", 1), ("flores_plus", 0),
    ("flores_plus", True), ("flores_plus", "1"), ("flores_plus", 1.5)])
def test_bad_identity_rejected(name, revision):
    with pytest.raises(ValueError):
        load(name, revision)


def test_absent_revision_never_falls_back():
    with pytest.raises(FileNotFoundError):
        load("unicode_emoji", 999)


@pytest.mark.parametrize("raw,values", [(b"$text\n", {}), (b"$text\n", {"text": "x", "extra": "y"}),
    (b"$text\n", {"text": 1}), (b"bad $\n", {})])
def test_invalid_substitution_rejected(raw, values):
    with pytest.raises(ValueError):
        PromptTemplate("control", 1, raw).render(**values)


def test_v2_changes_prompt_not_source_id_answer_or_verifier():
    groups = {"single-code-point": [{"sequence": "🙂", "name": "slightly smiling face"}]}
    a = list(emoji_tasks(groups, 1))[0]
    b = list(emoji_tasks(groups, 1, prompt_template=load("unicode_emoji", 2)))[0]
    assert a["prompt"] != b["prompt"]
    assert "literal characters" in b["prompt"]
    assert {k: v for k, v in a.items() if k != "prompt"} == {k: v for k, v in b.items() if k != "prompt"}
    assert digest(a) != digest(b)


def test_optional_metadata_is_backwards_compatible_and_hash_bound(tmp_path):
    manifest, _ = load_build(build(tmp_path))
    validate(manifest, "manifest")  # Old builds need no new field or rewrite.
    new = copy.deepcopy(manifest)
    new["source"]["prompt_templates"] = {"unicode_emoji": load("unicode_emoji").identity}
    validate(new, "manifest")
    assert digest(new) != digest(manifest)
    new["source"]["prompt_templates"]["unicode_emoji"] = load("unicode_emoji", 2).identity
    validate(new, "manifest")


@pytest.mark.parametrize("value", [{}, {"../escape": {"revision": 1, "sha256": "a" * 64}},
    {"control": {"revision": 0, "sha256": "a" * 64}},
    {"control": {"revision": True, "sha256": "a" * 64}},
    {"control": {"revision": 1, "sha256": "bad"}},
    {"control": {"revision": 1, "sha256": "a" * 64, "extra": 1}}])
def test_prompt_manifest_metadata_validation(tmp_path, value):
    manifest, _ = load_build(build(tmp_path))
    manifest["source"]["prompt_templates"] = value
    with pytest.raises(ValidationError):
        validate(manifest, "manifest")


@pytest.mark.parametrize("specifications", [["unknown=1"], ["unicode_emoji=0"], ["unicode_emoji=-1"],
    ["unicode_emoji=1.5"], ["unicode_emoji"], ["unicode_emoji=1", "unicode_emoji=2"]])
def test_invalid_cli_revisions_rejected(specifications):
    with pytest.raises(ValueError):
        parse_prompt_revisions(specifications)


def test_explicit_cli_revisions():
    assert parse_prompt_revisions([]) == {}
    assert parse_prompt_revisions(["unicode_emoji=2", "flores_plus=1"]) == {"unicode_emoji": 2, "flores_plus": 1}
