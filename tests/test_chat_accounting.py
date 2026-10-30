"""Chat count evidence is not a replacement for native positional evidence."""
import json
from copy import deepcopy

import pytest

from conftest import ByteVocabulary, record, sse
from murphy.client import completion_accounting, parse_body
from murphy.core.measure import inspect, numerical
from murphy.core.metrics import side_measurements
from scripts.report import reference_breakdown


def observed(stream=True, count=3, records=None, content="€", **fields):
    rows = [record(t) for t in (226, 130, 172)] if records is None else records
    choice = {"finish_reason": "stop", "logprobs": {"content": rows},
              "delta" if stream else "message": {"content": content, **fields}}
    payload = {"choices": [choice]}
    usage = {"completion_tokens": count}
    if stream:
        body = sse([payload, {"choices": [], "usage": usage}]) + b"data: [DONE]\n\n"
    else:
        body = json.dumps({**payload, "usage": usage}).encode()
    return {**parse_body(body, "/v1/chat/completions", stream),
            "mode": "stream" if stream else "nonstream",
            "condition": {"endpoint": "/v1/chat/completions"}}


@pytest.mark.parametrize('stream', [False, True])
def test_same_response_usage_and_vocabulary_are_complementary(stream):
    response = observed(stream)
    accounting = completion_accounting(response)
    assert accounting['count'] == 3
    assert accounting['scope'] == 'whole_response'
    checks = inspect(response, [response] * 2, ByteVocabulary())
    assert [c['status'] for c in checks] == ['pass', 'unavailable', 'pass']
    assert checks[0]['details']['count_delta'] == 0
    assert numerical([response] * 2, [response] * 2)['status'] == 'unavailable'
    measurements = side_measurements([response], [{'pass': True, 'reason': 'exact_match'}])
    assert measurements['token_level_coverage']['status'] == 'unavailable'
    assert measurements['actual_output_exposure']['status'] == 'unavailable'


@pytest.mark.parametrize('count,delta,reason', [(4, -1, 'record_count_shortfall'), (2, 1, 'record_count_excess')])
def test_counts_never_infer_positions(count, delta, reason):
    response = observed(count=count)
    check = inspect(response, [], ByteVocabulary())[0]
    assert check['status'] == 'fail'
    assert check['details']['count_delta'] == delta
    assert check['details']['failures'][0]['reason'] == reason
    assert 'positional_record_coverage' not in check['details']


def test_equal_count_does_not_prove_identities_or_bytes():
    rows = [record(226), record(226), record(172)]  # balanced omission/duplication
    response = observed(records=rows)
    assert inspect(response, [], ByteVocabulary())[0]['status'] == 'pass'
    assert inspect(response, [], ByteVocabulary())[1]['status'] == 'unavailable'
    response['chunks'][0]['records'][0]['bytes'] = [226, 130, 172]
    assert inspect(response, [], ByteVocabulary())[2]['status'] == 'fail'


@pytest.mark.parametrize('content,fields', [('', {}), ('A', {}), ('€', {'reasoning_content': 'hidden'})])
def test_presentation_is_not_the_record_denominator(content, fields):
    response = observed(content=content, **fields)
    assert inspect(response, [], ByteVocabulary())[0]['status'] == 'pass'


@pytest.mark.parametrize('bad', [True, -1, 1.5, '3', None, [], {}])
def test_malformed_usage_is_not_coerced(bad):
    response = observed(count=bad)
    assert completion_accounting(response)['reason'] == 'usage_malformed'
    assert inspect(response, [], ByteVocabulary())[0]['status'] == 'unavailable'


def test_missing_ambiguous_and_misplaced_usage():
    response = observed()
    usage = response['events'][1]
    response['events'][1]['payload']['usage'] = None
    assert completion_accounting(response)['reason'] == 'usage_missing'
    usage['payload']['usage'] = {'completion_tokens': 3}
    response['events'].insert(1, deepcopy(usage))
    assert completion_accounting(response)['reason'] == 'usage_ambiguous'
    response = observed()
    response['events'][0], response['events'][1] = response['events'][1], response['events'][0]
    assert completion_accounting(response)['reason'] == 'usage_not_final'


def test_intermediate_null_usage_and_terminal_incomplete():
    response = observed()
    response['events'][0]['payload']['usage'] = None
    assert completion_accounting(response)['count'] == 3
    response['status'] = 'incomplete'
    check = inspect(response, [], ByteVocabulary())[0]
    assert check['status'] == 'unavailable'
    assert check['details']['expected_records'] is None
    assert check['details']['reason'] == 'incomplete_transport'


def test_zero_count_is_a_valid_observation_not_a_coverage_fraction():
    response = observed(count=0, records=[], content='')
    assert inspect(response, [], ByteVocabulary())[0]['status'] == 'pass'


def test_report_never_cancels_shortfalls_with_excesses():
    checks = [inspect(observed(count=n), [], ByteVocabulary()) for n in (4, 2, 3)]
    rows = [{'condition_id': 'chat', 'mode': 'stream',
             **{side: {'checks_by_repeat': checks} for side in ('left', 'right')}}]
    groups = reference_breakdown(rows)
    group = next(g for g in groups if g['side'] == 'left' and g['category'] == 'missing_records')
    assert group['statuses'] == {'pass': 1, 'fail': 2}
    assert group['record_accounting'] == {
        'scope': 'whole_response', 'eligible_responses': 3, 'equal_count_responses': 1,
        'expected_records': 9, 'observed_records': 9, 'net_shortfall_sum': 1, 'net_excess_sum': 1}
    rows[0]['left']['checks_by_repeat'][0][0]['details']['count_delta'] = 0
    with pytest.raises(ValueError, match='Invalid chat record accounting'):
        reference_breakdown(rows)
