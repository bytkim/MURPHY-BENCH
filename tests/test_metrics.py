from copy import deepcopy

import pytest

from conftest import ByteVocabulary, record
from murphy.core.measure import inspect, numerical
from murphy.core.metrics import measurements, side_measurements


def grades(*passes):
    return [{"pass": value, "reason": "exact_match" if value else "answer_mismatch"} for value in passes]


def test_five_measurements_on_known_good_fixture(observed):
    rows = [observed, observed]
    result = measurements(rows, rows, grades(True, True), grades(True, True), numerical(rows, rows))
    assert set(result) == {"token_level_coverage", "complete_response_rate", "numerical_agreement", "actual_output_exposure", "task_compliance"}
    for name in ("token_level_coverage", "actual_output_exposure"):
        assert result[name]["left"]["numerator"] == result[name]["left"]["denominator"] == 8
        assert result[name]["left"]["value"] == 1
    assert result["complete_response_rate"]["left"]["value"] == result["task_compliance"]["left"]["value"] == 1
    # Four sampled positions x (selected logprob + one common candidate), not x repeats.
    assert result["numerical_agreement"]["numerator"] == result["numerical_agreement"]["denominator"] == 8
    assert result["numerical_agreement"]["value"] == 1
    assert result["numerical_agreement"]["details"]["accuracy_oracle"] is False


def test_missing_record_has_hand_counted_coverage_and_exposure(observed):
    bad = deepcopy(observed)
    bad["chunks"][1]["records"].pop()  # Last euro byte: 3/4 positions remain aligned.
    result = side_measurements([bad], grades(True))
    for name in ("token_level_coverage", "actual_output_exposure"):
        assert result[name]["numerator"] == 3
        assert result[name]["denominator"] == 4
        assert result[name]["value"] == 0.75


def test_exposure_primary_value_need_not_be_in_top_n(observed):
    for chunk in observed["chunks"]:
        for row in chunk["records"]:
            row["top_logprobs"] = [record(0)]
    result = side_measurements([observed], grades(True))["actual_output_exposure"]
    assert result["value"] == 1
    assert result["details"]["selected_field"] == 4
    assert result["details"]["topk_only"] == 0


def test_exposure_topk_only_and_absent_are_distinct(observed):
    bad = deepcopy(observed)
    del bad["chunks"][0]["records"][0]["logprob"]
    result = side_measurements([bad], grades(True))["actual_output_exposure"]
    assert result["value"] == 1
    assert result["details"]["selected_field"] == 3
    assert result["details"]["topk_only"] == 1
    assert numerical([bad] * 2, [bad] * 2)["status"] == "fail"  # No selected-field repair.
    bad["chunks"][0]["records"][0]["top_logprobs"] = []
    result = side_measurements([bad], grades(True))["actual_output_exposure"]
    assert result["value"] == 0.75
    assert result["details"]["unexposed"] == 1


@pytest.mark.parametrize("value", [None, True, "-1", float("nan"), float("inf")])
def test_null_or_nonfinite_is_not_exposed(observed, value):
    row = observed["chunks"][0]["records"][0]
    row["logprob"], row["top_logprobs"] = value, []
    assert side_measurements([observed], grades(True))["actual_output_exposure"]["value"] == 0.75


def test_joint_loss_does_not_invent_missing_ids(observed):
    bad = deepcopy(observed)
    bad["chunks"][1]["records"].pop()
    bad["chunks"][1]["token_ids"].pop()
    result = side_measurements([bad], grades(True))
    assert result["token_level_coverage"]["value"] == 1
    assert result["token_level_coverage"]["denominator"] == 3
    assert inspect(bad, [observed] * 2, ByteVocabulary())[1]["status"] == "fail"


def test_completion_and_answer_accuracy_have_different_denominators(observed):
    incomplete = deepcopy(observed)
    incomplete["status"] = "incomplete"
    result = side_measurements([observed, observed, incomplete], [*grades(True, False), {"pass": None, "reason": "incomplete_transport"}])
    assert result["complete_response_rate"]["value"] == pytest.approx(2 / 3)
    assert result["task_compliance"]["value"] == 0.5
    assert result["task_compliance"]["denominator"] == 2
    assert result["task_compliance"]["status"] == "partial"
    assert result["token_level_coverage"]["status"] == "partial"


def test_stop_or_length_finish_is_complete_but_not_graded(observed):
    result = side_measurements([observed], [{"pass": None, "reason": "intentional_condition"}])
    assert result["complete_response_rate"]["value"] == 1
    assert result["task_compliance"]["denominator"] == 0
    assert result["task_compliance"]["value"] is None


def test_chat_and_empty_native_output_never_invent_denominators(observed):
    for chunk in observed["chunks"]:
        chunk["token_ids"] = None
    result = side_measurements([observed], grades(True))
    for name in ("token_level_coverage", "actual_output_exposure"):
        assert result[name]["status"] == "unavailable"
        assert result[name]["denominator"] is None and result[name]["value"] is None
    for chunk in observed["chunks"]:
        chunk["token_ids"], chunk["records"] = [], []
    result = side_measurements([observed], grades(True))["actual_output_exposure"]
    assert result["denominator"] == 0 and result["value"] is None


def test_numerical_measurement_has_known_disagreement_fraction(observed):
    bad = deepcopy(observed)
    bad["chunks"][0]["records"][0]["logprob"] = -0.2
    result = measurements([observed] * 2, [bad] * 2, grades(True, True), grades(True, True), numerical([observed] * 2, [bad] * 2))
    metric = result["numerical_agreement"]
    assert metric["numerator"] == 7 and metric["denominator"] == 8
    assert metric["value"] == 0.875
    assert metric["details"]["max_absolute_delta"] == pytest.approx(0.1)


def test_numerical_missing_reference_is_null_not_zero_or_pass(observed):
    result = measurements([observed], [observed], grades(True), grades(True), numerical([observed], [observed]))
    assert result["numerical_agreement"]["value"] is None
    assert result["numerical_agreement"]["denominator"] is None


def test_repeated_token_positions_count_separately(observed):
    observed["chunks"] = [{"token_ids": [65, 65], "records": [record(65), record(65)],
                           "content": "AA", "reasoning_content": "", "finish": "eos"}]
    result = side_measurements([observed], grades(True))
    assert result["actual_output_exposure"]["denominator"] == 2
    assert result["actual_output_exposure"]["numerator"] == 2
    observed["chunks"][0]["records"].pop()
    assert side_measurements([observed], grades(True))["actual_output_exposure"]["value"] == 0.5


def test_exposure_is_not_probability_accuracy(observed):
    observed["chunks"][0]["records"][0]["logprob"] = 5.0
    assert side_measurements([observed], grades(True))["actual_output_exposure"]["value"] == 1
    assert numerical([observed] * 2, [observed] * 2)["status"] == "fail"


def test_missing_only_one_chunks_ids_is_partial(observed):
    observed["chunks"][0]["token_ids"] = None
    result = side_measurements([observed], grades(True))["actual_output_exposure"]
    assert result["status"] == "partial" and result["denominator"] == 3
    assert result["details"]["unknown_token_chunks"] == 1
