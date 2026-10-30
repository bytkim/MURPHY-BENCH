from copy import deepcopy
import pytest

from conftest import ByteVocabulary
from murphy.core.measure import inspect, numerical, stability


def test_clean_and_missing_record(observed):
    assert all(c["status"] == "pass" for c in inspect(observed, [observed, observed], ByteVocabulary()))
    bad = deepcopy(observed)
    bad["chunks"][1]["records"].pop(0)
    checks = inspect(bad, [observed, observed], ByteVocabulary())
    assert checks[0]["status"] == checks[1]["status"] == "fail"
    assert checks[2]["status"] == "pass"  # Surviving byte labels are not thereby wrong.


def test_joint_omission_needs_nonstream_reference(observed):
    bad = deepcopy(observed)
    bad["chunks"][1]["records"].pop(0)
    bad["chunks"][1]["token_ids"].pop(0)
    checks = inspect(bad, [observed, observed], ByteVocabulary())
    assert checks[0]["status"] == "pass"
    assert checks[1]["status"] == "fail"


def test_wrong_bytes_duplicate_and_reordered_ids(observed):
    bad = deepcopy(observed)
    bad["chunks"][1]["records"][0]["bytes"] = [0]
    assert inspect(bad, [observed, observed], ByteVocabulary())[2]["status"] == "fail"
    for tokens in ([226, 226, 172], [172, 130, 226]):
        bad = deepcopy(observed)
        bad["chunks"][1]["token_ids"] = tokens
        assert inspect(bad, [observed, observed], ByteVocabulary())[1]["status"] == "fail"


def test_numerical_drift_and_different_context(observed):
    assert numerical([observed] * 2, [observed] * 2)["status"] == "pass"
    bad = deepcopy(observed)
    bad["chunks"][0]["records"][0]["logprob"] += 0.1
    assert numerical([observed] * 2, [bad] * 2)["status"] == "fail"
    assert numerical([observed, bad], [bad] * 2)["status"] == "unavailable"
    bad["chunks"][0]["token_ids"] = [66]
    assert numerical([observed] * 2, [bad] * 2)["status"] == "unavailable"


def test_candidate_set_disagreement_and_partial_capture(observed):
    bad = deepcopy(observed)
    bad["chunks"][0]["records"][0]["top_logprobs"] = []
    assert numerical([observed] * 2, [bad] * 2)["status"] == "fail"
    bad["status"] = "incomplete"
    assert not stability([bad] * 2)
    assert all(c["status"] == "unavailable" for c in inspect(bad, [observed] * 2, ByteVocabulary()))


def test_known_local_failure_not_hidden_by_unavailable_reference(observed):
    bad = deepcopy(observed)
    bad["chunks"][0]["records"][0]["id"] = 66
    checks = inspect(bad, [], ByteVocabulary())
    assert checks[1]["status"] == "fail"
    assert checks[1]["details"]["nonstream_reference_checked"] is False
    assert checks[1]["details"]["reason"] == "stable_native_reference_unavailable"


def test_id_reference_does_not_require_bit_identical_probabilities(observed):
    other = deepcopy(observed)
    other["chunks"][0]["records"][0]["logprob"] -= 0.1
    assert not stability([observed, other])
    assert inspect(observed, [observed, other], ByteVocabulary())[1]["status"] == "pass"


@pytest.mark.parametrize("bad_id", [True, -1, 1.5, "65", None])
def test_wrong_token_identity_rejects_invalid_types(observed, bad_id):
    observed["chunks"][0]["token_ids"] = [bad_id]
    assert inspect(observed, [observed] * 2, ByteVocabulary())[1]["status"] == "fail"
    assert numerical([observed] * 2, [observed] * 2)["status"] == "unavailable"


def test_missing_required_native_array_is_failure_not_capability_exclusion(observed):
    observed["condition"] = {"endpoint": "/completion"}
    observed["chunks"][0]["token_ids"] = None
    assert inspect(observed, [], ByteVocabulary())[0]["status"] == "fail"


@pytest.mark.parametrize("bad_bytes", [None, [True], [256], [-1], "A", [65.0]])
def test_byte_labels_are_raw_octets_not_coerced_text(observed, bad_bytes):
    observed["chunks"][0]["records"][0]["bytes"] = bad_bytes
    assert inspect(observed, [observed] * 2, ByteVocabulary())[2]["status"] == "fail"


def test_top_candidate_byte_labels_are_checked_too(observed):
    observed["chunks"][0]["records"][0]["top_logprobs"][0]["bytes"] = [0]
    failures = inspect(observed, [observed] * 2, ByteVocabulary())[2]["details"]["failures"]
    assert len(failures) == 1 and failures[0]["candidate"] == 1


@pytest.mark.parametrize("value", [None, True, "-1", 0.1, float("inf"), float("nan"), -(10 ** 1000)])
def test_numerical_invalid_values_do_not_become_agreement(observed, value):
    observed["chunks"][0]["records"][0]["logprob"] = value
    result = numerical([observed] * 2, [observed] * 2)
    assert result["status"] == "fail"
    assert result["details"]["invalid_values"] == 1


@pytest.mark.parametrize("delta,status", [(1e-7, "pass"), (1e-4, "fail")])
def test_numerical_tolerance(observed, delta, status):
    other = deepcopy(observed)
    other["chunks"][0]["records"][0]["logprob"] += delta
    assert numerical([observed] * 2, [other] * 2)["status"] == status


def test_candidate_differences_do_not_impute_zero(observed):
    other = deepcopy(observed)
    other["chunks"][0]["records"][0]["top_logprobs"] = []
    result = numerical([observed] * 2, [other] * 2)
    assert result["status"] == "fail"
    assert result["details"]["compared_values"] == 7
    assert result["details"]["agreeing_values"] == 7
    assert result["details"]["candidate_set_differences"] == [{"position": 0, "left_only": [65], "right_only": []}]


def test_duplicate_top_candidate_identity_is_not_collapsed_silently(observed):
    candidates = observed["chunks"][0]["records"][0]["top_logprobs"]
    candidates.append(deepcopy(candidates[0]))
    assert numerical([observed] * 2, [observed] * 2)["status"] == "fail"


def test_matching_invalid_ids_do_not_count_as_aligned(observed):
    observed["chunks"][0]["token_ids"] = [-1]
    observed["chunks"][0]["records"][0]["id"] = -1
    check = inspect(observed, [observed] * 2, ByteVocabulary())[0]
    assert check["details"]["aligned_record_positions"] == 3
