"""Configuration-only controls; no server, GPU or inference is used."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
from jsonschema import ValidationError

from adapters.core.policy import DEFAULT, check, load
from murphy.core.run import request_for
from murphy.util.io import load_build, validate

PUBLIC = Path(os.environ['MURPHY_SOURCE'])


def test_v4_changes_only_endpoint_selection_and_recipe_revision():
    previous, _ = load(PUBLIC / 'policies/murphy.v3.json')
    paired, _ = load(PUBLIC / 'policies/murphy.v4.json')
    expected = deepcopy(previous)
    expected['revision'] = 4
    expected['request']['endpoints'] = ['/completion', '/v1/chat/completions']
    assert paired == expected
    assert DEFAULT == PUBLIC / 'policies/murphy.v1.json'


@pytest.mark.parametrize('revision', [1, 2, 3])
def test_legacy_policies_remain_native_only_without_mutation(revision):
    path = PUBLIC / f'policies/murphy.v{revision}.json'
    policy, _ = load(path)
    assert 'endpoints' not in policy['request']
    assert policy == json.loads(path.read_text())


@pytest.mark.parametrize('endpoints', [[], ['/completion', '/completion'],
    ['/v1/completions'], ['/completion', '/v1/chat/completions', '/completion'],
    '/completion', None, [True]])
def test_invalid_endpoint_configuration_rejected(endpoints):
    policy, _ = load(PUBLIC / 'policies/murphy.v3.json')
    policy['request']['endpoints'] = endpoints
    with pytest.raises(ValidationError):
        check(policy)


@pytest.mark.parametrize('endpoints', [['/completion'], ['/v1/chat/completions'],
    ['/completion', '/v1/chat/completions'], ['/v1/chat/completions', '/completion']])
def test_supported_endpoint_order_is_explicit(endpoints):
    policy, _ = load(PUBLIC / 'policies/murphy.v3.json')
    policy['request']['endpoints'] = endpoints
    assert check(policy)['request']['endpoints'] == endpoints


@pytest.mark.parametrize('mode', ['nonstream', 'stream'])
def test_requests_share_controls_but_not_transport_fields(mode):
    common = {'task_ids': ['control'], 'stop': [], 'max_tokens': 384,
              'reasoning': 'off', 'literal_grammar': False, 'grade': True}
    native = {'id': 'native-nonthinking', **common, 'endpoint': '/completion',
              'prompt_wrapper': {'prefix': 'USER:', 'suffix': '\nASSISTANT:'}}
    chat = {'id': 'chat-nonthinking', **common, 'endpoint': '/v1/chat/completions'}
    task = {'prompt': 'Repeat 🙂 exactly.', 'expected': '🙂'}
    a, b = (request_for(task, c, mode) for c in (native, chat))
    for key in ('temperature', 'seed', 'cache_prompt', 'return_tokens', 'stream', 'stop'):
        assert a[key] == b[key]
    assert a['n_predict'] == b['max_tokens'] == 384
    assert a['n_probs'] == b['top_logprobs'] == 5
    assert a['prompt'] == 'USER:' + b['messages'][0]['content'] + '\nASSISTANT:'
    assert b['messages'] == [{'role': 'user', 'content': task['prompt']}]
    assert b['chat_template_kwargs'] == {'enable_thinking': False}
    assert 'messages' not in a and 'prompt' not in b
    assert 'grammar' not in a and 'grammar' not in b
    assert b.get('stream_options') == ({'include_usage': True} if mode == 'stream' else None)


def test_experiment_schema_does_not_override_build_conditions():
    recipe = {'schema_version': '1', 'experiment_id': 'configuration-control',
              'builds': [{'path': 'manifest.json', 'sha256': '0' * 64}],
              'targets': [{'path': name, 'sha256': '0' * 64} for name in ('baseline.toml', 'patched.toml')],
              'expected_requests': 24000, 'timeout_seconds': 36000}
    validate(recipe, 'experiment')
    with pytest.raises(ValidationError):
        validate({**recipe, 'endpoints': ['/completion']}, 'experiment')
