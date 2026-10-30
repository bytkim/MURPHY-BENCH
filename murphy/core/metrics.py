"""Five measurements, with explicit denominators and no new failure categories."""
from murphy.core.measure import finite_number, token_id


def rate(reference, numerator, denominator, *, partial=False, reason=None, **details):
    available = denominator is not None and denominator > 0
    if reason:
        details["reason"] = reason
    return {"status": ("partial" if partial else "available") if available else "unavailable",
            "reference": reference, "numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if available else None, "details": details}


def token_observations(responses):
    counts = {"tokens": 0, "aligned_records": 0, "selected_field": 0, "topk_only": 0,
              "unexposed": 0, "invalid_token_ids": 0, "unknown_token_chunks": 0,
              "incomplete_responses": 0, "known_token_chunks": 0}
    for response in responses:
        counts["incomplete_responses"] += response["status"] != "complete"
        for chunk in response["chunks"]:
            ids, rows = chunk["token_ids"], chunk["records"] or []
            if ids is None:
                counts["unknown_token_chunks"] += 1
                continue
            counts["known_token_chunks"] += 1
            counts["tokens"] += len(ids)
            for position, emitted in enumerate(ids):
                if not token_id(emitted):
                    counts["invalid_token_ids"] += 1
                row = rows[position] if position < len(rows) else None
                aligned = (token_id(emitted) and isinstance(row, dict) and
                           token_id(row.get("id")) and row["id"] == emitted)
                counts["aligned_records"] += aligned
                if aligned and finite_number(row.get("logprob")):
                    counts["selected_field"] += 1
                    continue
                # An actual-token candidate can expose the value, but does not repair
                # a missing selected field. Require an unambiguous aligned position.
                candidates = row.get("top_logprobs") if aligned else None
                own = [c for c in candidates if isinstance(c, dict) and token_id(c.get("id")) and c["id"] == emitted] if isinstance(candidates, list) else []
                if len(own) == 1 and finite_number(own[0].get("logprob")):
                    counts["topk_only"] += 1
                else:
                    counts["unexposed"] += 1
    return counts


def side_measurements(responses, grades):
    if len(responses) != len(grades):
        raise ValueError("Each attempt needs exactly one task grade")
    counts = token_observations(responses)
    known = counts["known_token_chunks"] > 0
    denominator = counts["tokens"] if known else None
    partial = bool(counts["unknown_token_chunks"] or counts["incomplete_responses"] or counts["invalid_token_ids"])
    reference = "observed_same_chunk_native_token_positions"
    reason = None if known else "native_token_reference_unavailable"
    graded = [g for g in grades if g["pass"] is not None]
    return {
        "token_level_coverage": rate(reference, counts["aligned_records"], denominator,
                                     partial=partial, reason=reason, **counts),
        "complete_response_rate": rate("all_manifest_planned_attempts", sum(r["status"] == "complete" for r in responses),
                                       len(responses), incomplete_attempts=sum(r["status"] != "complete" for r in responses)),
        "actual_output_exposure": rate(reference, counts["selected_field"] + counts["topk_only"], denominator,
                                       partial=partial, reason=reason, **counts),
        "task_compliance": rate("task_expected_answer_and_declared_verifier", sum(g["pass"] is True for g in graded), len(graded),
                                partial=any(g["reason"] == "incomplete_transport" for g in grades),
                                graded_attempts=len(graded), wrong_answers=sum(g["pass"] is False for g in graded),
                                intentional_conditions=sum(g["reason"] == "intentional_condition" for g in grades),
                                incomplete_attempts=sum(g["reason"] == "incomplete_transport" for g in grades)),
    }


def measurements(left, right, left_grades, right_grades, numerical_check):
    a, b = side_measurements(left, left_grades), side_measurements(right, right_grades)
    result = {name: {"left": a[name], "right": b[name]} for name in a}
    d = numerical_check["details"]
    eligible = numerical_check["status"] != "unavailable"
    invalid = d.get("invalid_values", 0)
    support = len(d.get("candidate_set_differences", []))
    result["numerical_agreement"] = rate("other_build_same_mode_aligned_logprob_values",
        d.get("agreeing_values", 0), d.get("compared_values", 0) if eligible else None,
        partial=bool(invalid or support or any("reason" in f for f in d.get("failures", []))),
        reason=d.get("reason"), invalid_values=invalid, candidate_set_difference_positions=support,
        max_absolute_delta=d.get("max_absolute_delta") if eligible else None,
        atol=d.get("atol"), rtol=d.get("rtol"), accuracy_oracle=False)
    return result
