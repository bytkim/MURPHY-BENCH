import copy
from adapters.core.prepare import prepare
import json

import pytest
from jsonschema import ValidationError

from adapters.util.fixture import build
from adapters.core.policy import DEFAULT, allocation, check, load
from adapters.core.common import aligned_ids
from murphy.util.io import digest, file_hash, load_build


def test_policy_snapshot_and_derived_allocations():
    policy, identity = load()
    assert identity["sha256"] == file_hash(DEFAULT)
    assert identity["snapshot_sha256"] == digest(policy)
    assert identity["snapshot"] == policy
    assert allocation(policy, "trial") == {"flores_plus": 18, "belebele": 12, "unicode_emoji": 20}
    assert allocation(policy, "full") == {"flores_plus": 900, "belebele": 300, "unicode_emoji": 300}


@pytest.mark.parametrize("path,value", [
    (("profiles", "trial", "emoji_tasks"), 21),
    (("profiles", "trial", "flores_ids"), 151),
    (("profiles", "full", "emoji_tasks"), 305),
    (("profiles", "full", "belebele_ids"), 1),
    (("datasets", "unicode_emoji", "grouping", "strategy"), "unknown"),
    (("selection", "seed"), True),
    (("languages", "fixed"), ["eng_Latn", "eng_Latn"]),
    (("revision",), 0),
    (("unknown",), "no silent fields"),
])
def test_invalid_recipe_fails_closed(path, value):
    policy, _ = load()
    current = policy
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    with pytest.raises((ValueError, ValidationError)):
        check(policy)


def test_selection_seed_is_an_input_not_a_hidden_constant():
    tables = {"a": {i: {} for i in range(100)}}
    assert aligned_ids(tables, 20, "control") == aligned_ids(tables, 20, "control", seed=42)
    assert aligned_ids(tables, 20, "control", seed=43) != aligned_ids(tables, 20, "control", seed=42)


def test_policy_explicit_unknown_profile_and_conflicting_prompt_fail_before_io(tmp_path):
    with pytest.raises(ValueError, match="profile"):
        prepare(tmp_path, tmp_path, tmp_path, profile="unknown")
    with pytest.raises(ValueError, match="match the explicit policy"):
        prepare(tmp_path, tmp_path, tmp_path, prompt_revisions={"unicode_emoji": 2})


def test_historical_build_readable_and_snapshot_tampering_rejected(tmp_path):
    destination = build(tmp_path)
    manifest, _ = load_build(destination)
    _, identity = load()
    manifest["source"]["policy"] = identity
    path = destination / "manifest.json"
    path.write_text(json.dumps(manifest))
    load_build(destination)
    changed = copy.deepcopy(manifest)
    changed["source"]["policy"]["snapshot"]["revision"] = 2
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Policy snapshot"):
        load_build(destination)
