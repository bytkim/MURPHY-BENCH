"""Four independent checks; absent references produce unavailable, never pass."""
import math

from murphy.client import completion_accounting, joined


def check(category, reference, *, failures=None, unavailable=None, **details):
    details.update(failures=failures or [])
    if unavailable:
        details["reason"] = unavailable
    return {"category": category, "status": "fail" if failures else ("unavailable" if unavailable else "pass"),
            "reference": reference, "details": details}


def token_id(value):
    return type(value) is int and value >= 0


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def token_reference_ready(responses):
    if len(responses) < 2 or any(r["status"] != "complete" for r in responses):
        return False
    ids = [joined(r)["token_ids"] for r in responses]
    return all(row is not None and all(token_id(i) for i in row) and row == ids[0] for row in ids)


def stability(responses):
    if len(responses) < 2 or any(r["status"] != "complete" for r in responses):
        return False
    views = [joined(r) for r in responses]
    return all(view == views[0] for view in views[1:])


def inspect(response, nonstream_repeats, vocabulary):
    view = joined(response)
    missing, identities, byte_failures = [], [], []
    tokens_n, records_n, bytes_n, aligned_n = 0, 0, 0, 0
    for index, chunk in enumerate(response["chunks"]):
        ids, rows = chunk["token_ids"], chunk["records"] or []
        records_n += len(rows)
        if ids is not None:
            tokens_n += len(ids)
            if any(not token_id(i) for i in ids):
                identities.append({"chunk": index, "reason": "invalid_native_token_ids"})
            if len(ids) != len(rows):
                missing.append({"chunk": index, "token_count": len(ids), "record_count": len(rows)})
            row_ids = [r.get("id") if isinstance(r, dict) else None for r in rows]
            aligned_n += sum(token_id(a) and token_id(b) and a == b for a, b in zip(ids, row_ids))
            if any(not token_id(i) for i in row_ids) or row_ids != ids:
                identities.append({"chunk": index, "expected_ids": ids, "record_ids": row_ids})
        elif response.get("condition", {}).get("endpoint") == "/completion":
            missing.append({"chunk": index, "reason": "required_native_token_array_missing"})
        for ri, row in enumerate(rows):
            if not isinstance(row, dict):
                byte_failures.append({"chunk": index, "record": ri, "reason": "malformed_record"})
                continue
            candidates = row.get("top_logprobs", [])
            if not isinstance(candidates, list):
                candidates = []
                byte_failures.append({"chunk": index, "record": ri, "reason": "malformed_top_candidates"})
            for ci, record in enumerate([row, *candidates]):
                bytes_n += 1
                try:
                    expected = vocabulary.token_bytes(record.get("id"))
                    actual = record.get("bytes")
                    valid = isinstance(actual, list) and all(type(b) is int and 0 <= b <= 255 for b in actual)
                    if not valid or actual != expected:
                        byte_failures.append({"chunk": index, "record": ri, "candidate": ci,
                                              "id": record.get("id"), "expected": expected, "actual": actual})
                except (ValueError, AttributeError) as exc:
                    byte_failures.append({"chunk": index, "record": ri, "candidate": ci, "reason": str(exc)})
    native = view["token_ids"] is not None
    reference_ready = token_reference_ready(nonstream_repeats)
    if native and reference_ready:
        expected_ids = joined(nonstream_repeats[0])["token_ids"]
        if view["token_ids"] != expected_ids:
            identities.append({"scope": "joined_vs_nonstream", "expected_ids": expected_ids, "actual_ids": view["token_ids"]})
    transport = None if response["status"] == "complete" else "incomplete_transport"
    count_check = check("missing_records", "same_chunk_native_token_ids", failures=missing,
              unavailable=transport or (None if native else "native_ids_not_exposed"), tokens=tokens_n if native else None,
              records=records_n, aligned_record_positions=aligned_n if native else None,
              positional_record_coverage=aligned_n / tokens_n if native and tokens_n else None)
    if response.get("condition", {}).get("endpoint") == "/v1/chat/completions":
        accounting = completion_accounting(response)
        expected = accounting["count"]
        delta = records_n - expected if expected is not None else None
        failures = [] if delta in (None, 0) else [{
            "reason": "record_count_shortfall" if delta < 0 else "record_count_excess",
            "expected_records": expected, "observed_records": records_n, "count_delta": delta}]
        count_check = check("missing_records", accounting["reference"], failures=failures,
                            unavailable=accounting["reason"], scope=accounting["scope"],
                            usage_event_index=accounting["event_index"], expected_records=expected,
                            observed_records=records_n, count_delta=delta)
    return [
        count_check,
        check("wrong_token_identities", "stable_same_build_nonstream_and_chunk_ids", failures=identities,
              unavailable=transport or (None if native and reference_ready else "stable_native_reference_unavailable"),
              nonstream_reference_checked=bool(native and reference_ready)),
        check("byte_label_mismatches", "independent_gguf_vocabulary_special_false", failures=byte_failures,
              unavailable=transport or (None if bytes_n else "no_records_to_check"), checked_records=bytes_n),
    ]


def numerical(left_repeats, right_repeats, *, atol=1e-6, rtol=1e-5):
    if not finite_number(atol) or not finite_number(rtol) or atol < 0 or rtol < 0:
        raise ValueError("Numerical tolerances must be finite and nonnegative")
    reference = "other_build_same_mode_stable_token_history"
    if not stability(left_repeats) or not stability(right_repeats):
        return check("numerical_disagreement", reference, unavailable="incomplete_or_unstable_repeats")
    left, right = joined(left_repeats[0]), joined(right_repeats[0])
    # Native IDs are needed to prove complete aligned sampled histories.
    if left["token_ids"] is None or right["token_ids"] is None:
        return check("numerical_disagreement", reference, unavailable="native_history_not_exposed")
    for view in (left, right):
        ids = [r.get("id") if isinstance(r, dict) else None for r in view["records"]]
        if any(not token_id(i) for i in [*ids, *view["token_ids"]]) or ids != view["token_ids"]:
            return check("numerical_disagreement", reference, unavailable="records_not_aligned_with_sampled_history")
    if left["token_ids"] != right["token_ids"] or not left["records"]:
        return check("numerical_disagreement", reference, unavailable="different_or_empty_token_history")
    failures, deltas, support = [], [], []
    def compare_value(a, b, position, candidate):
        if not all(finite_number(x) and x <= 0 for x in (a, b)):
            failures.append({"position": position, "candidate": candidate, "reason": "invalid_or_missing_logprob"})
            return
        delta = abs(a - b)
        agrees = math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
        deltas.append({"position": position, "candidate": candidate, "left": a, "right": b,
                       "absolute_delta": delta, "within_tolerance": agrees})
        if not agrees:
            failures.append(deltas[-1])
    for i, (a, b) in enumerate(zip(left["records"], right["records"])):
        compare_value(a.get("logprob"), b.get("logprob"), i, "selected")
        maps = []
        for row in (a, b):
            candidates = row.get("top_logprobs")
            if not isinstance(candidates, list) or any(not isinstance(c, dict) or not token_id(c.get("id")) for c in candidates):
                failures.append({"position": i, "reason": "malformed_top_candidates"})
                candidates = []
            mapped = {c["id"]: c for c in candidates}
            if len(mapped) != len(candidates):
                failures.append({"position": i, "reason": "duplicate_top_candidate_ids"})
            maps.append(mapped)
        aa, bb = maps
        if aa.keys() != bb.keys():
            difference = {"position": i, "left_only": sorted(aa.keys() - bb.keys()), "right_only": sorted(bb.keys() - aa.keys())}
            support.append(difference)
            failures.append({"reason": "candidate_set_difference", **difference})
        for token in sorted(aa.keys() & bb.keys()):
            compare_value(aa[token].get("logprob"), bb[token].get("logprob"), i, token)
    return check("numerical_disagreement", reference, failures=failures, atol=atol, rtol=rtol,
                 compared_values=len(deltas), max_absolute_delta=max((d["absolute_delta"] for d in deltas), default=0),
                 agreeing_values=sum(d["within_tolerance"] for d in deltas),
                 invalid_values=sum(f.get("reason") == "invalid_or_missing_logprob" for f in failures),
                 deltas=deltas, candidate_set_differences=support)
