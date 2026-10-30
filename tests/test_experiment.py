"""Composition tests use existing pipeline controls, never alternative scoring."""
import json
import os
import shutil
import socket
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

from test_pipeline import pipeline, make_pipeline
from murphy.core.compare import compare
from murphy.util.io import file_hash, load_build, read_jsonl, write_json
from scripts import evaluate, report


def test_chat_accounting_through_existing_pipeline_and_offline_report(tmp_path, monkeypatch):
    from test_pipeline import make_pipeline
    completion, path = completed(make_pipeline(tmp_path, monkeypatch, chat_usage=True), tmp_path)
    summary = report.summarize(completion)
    assert summary['schema_version'] == '3'
    rows = read_jsonl(path, 'comparison.v2')
    chat = [r for r in rows if r['condition_id'] == 'chat']
    assert all(r['measurements']['token_level_coverage']['right']['status'] == 'unavailable' for r in chat)
    assert all(r['numerical']['status'] == 'unavailable' for r in chat)
    groups = summary['datasets'][0]['overall']['reference_breakdown']
    counts = [g for g in groups if g['reference'] == 'chat_usage_completion_tokens']
    assert {g['mode'] for g in counts} == {'stream', 'nonstream'}
    assert any(g['statuses'].get('fail') for g in counts if g['side'] == 'left')
    assert all(not g['statuses'].get('fail') for g in counts if g['side'] == 'right')
    first, second = tmp_path / 'chat-report1', tmp_path / 'chat-report2'
    report.report(completion, first)
    report.report(completion, second)
    for name in ('summary.json', 'report.rst'):
        assert (first / name).read_bytes() == (second / name).read_bytes()


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    from adapters.util.fixture import build
    from conftest import ByteVocabulary
    directory = build(tmp_path / 'data')
    model, binary, template = (tmp_path / n for n in ('model', 'binary', 'template'))
    for p in (model, binary, template):
        p.write_text('unit test only')
    targets = []
    for lane in ('baseline', 'candidate'):
        p = tmp_path / (lane + '.toml')
        p.write_text(f'''schema_version="1"
target_id="{lane}"
base_url="http://127.0.0.1:18783"
model_path="model"
binary_path="binary"
model_sha256="{file_hash(model)}"
binary_sha256="{file_hash(binary)}"
source_commit="{'a'*40}"
reasoning_capable=true
[runtime_config]
backend="cpu"
threads=1
parallel=1
special=false
backend_sampling=false
context_size=512
batch_size=32
ubatch_size=32
gpu_layers=0
flash_attention="off"
''')
        targets.append({'path': p.name, 'sha256': file_hash(p)})
    value = {'schema_version':'1', 'experiment_id':'fixture',
             'builds':[{'path':str(directory/'manifest.json'), 'sha256':file_hash(directory/'manifest.json')}],
             'targets':targets, 'template':{'path':template.name, 'sha256':file_hash(template)},
             'expected_requests':128, 'timeout_seconds':30}
    path = tmp_path / 'recipe.json'
    write_json(path, value, 'experiment')
    monkeypatch.setattr(evaluate, 'Vocabulary', lambda _: ByteVocabulary())
    return path


def test_prepare_is_offline_and_relative_to_recipe(recipe, monkeypatch, tmp_path):
    monkeypatch.chdir('/')
    monkeypatch.setattr(evaluate, 'Client', lambda _: pytest.fail('Offline admission contacted server'))
    value, builds, targets, template = evaluate.prepare(recipe)
    assert value['expected_requests'] == 128
    assert len(builds) == 1 and len(targets) == 2
    assert template == tmp_path / 'template'
    assert targets[0]['model_path'] == str(tmp_path / 'model')
    assert not (builds[0][0] / 'runs').exists()


@pytest.mark.parametrize('mutation', ['count', 'duplicate_build', 'target_hash', 'model_hash',
                                     'existing_run', 'template_hash', 'tasks_hash'])
def test_prepare_rejects_before_launch(recipe, mutation, tmp_path):
    value = json.loads(recipe.read_text())
    if mutation == 'count':
        value['expected_requests'] += 1
    elif mutation == 'duplicate_build':
        value['builds'] *= 2
    elif mutation == 'target_hash':
        value['targets'][0]['sha256'] = '0' * 64
    elif mutation == 'model_hash':
        (tmp_path / 'model').write_text('changed')
    elif mutation == 'existing_run':
        (Path(value['builds'][0]['path']).parent / 'runs/baseline').mkdir(parents=True)
    elif mutation == 'template_hash':
        (tmp_path / 'template').write_text('changed')
    else:
        (Path(value['builds'][0]['path']).parent / 'tasks.jsonl').write_text('')
    recipe.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileExistsError)):
        evaluate.prepare(recipe)


def test_owned_server_refuses_occupied_port(recipe, tmp_path):
    _, _, targets, template = evaluate.prepare(recipe)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        targets[0]['base_url'] = f'http://127.0.0.1:{sock.getsockname()[1]}'
        with pytest.raises(OSError):
            with evaluate.server(targets[0], template, tmp_path):
                pytest.fail('Attached to occupied port')


@pytest.mark.parametrize('during_readiness', [False, True])
def test_owned_child_stopped_on_failure_or_interrupt(recipe, tmp_path, monkeypatch, during_readiness):
    _, _, targets, template = evaluate.prepare(recipe)
    events = []
    class Process:
        pid = 12345
        returncode = None
        def poll(self):
            return self.returncode
        def terminate(self):
            events.append('terminate')
        def wait(self, timeout):
            self.returncode = 0
    class Client:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def properties(self):
            if during_readiness:
                raise evaluate.ExperimentInterrupted('deadline')
            return {'build_info':'a'*40, 'total_slots':1}
    monkeypatch.setattr(evaluate.subprocess, 'Popen', lambda *a, **k: Process())
    monkeypatch.setattr(evaluate, 'Client', lambda *a, **k: Client())
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        targets[0]['base_url'] = f'http://127.0.0.1:{sock.getsockname()[1]}'
    with pytest.raises(evaluate.ExperimentInterrupted if during_readiness else RuntimeError):
        with evaluate.server(targets[0], template, tmp_path):
            raise RuntimeError('body')
    assert events == ['terminate']
    assert json.loads((tmp_path/'baseline-shutdown.json').read_text())['data']['owned_child_stopped']


def test_metric_aggregation_preserves_missing_and_uses_counts():
    metric = report.fractions([{'status':'available','numerator':1,'denominator':2},
                               {'status':'partial','numerator':9,'denominator':10},
                               {'status':'unavailable','numerator':0,'denominator':None}])
    assert metric['value'] == 10/12
    assert metric['group_statuses'] == {'available':1,'partial':1,'unavailable':1}
    assert report.fractions([{'status':'unavailable','numerator':0,'denominator':None}])['value'] is None


def completed(pipeline, tmp_path):
    build, good, bad = pipeline
    path = compare(build, bad, good)
    rows = read_jsonl(path, 'comparison.v2')
    output = tmp_path / 'experiment'
    output.mkdir()
    recipe = {'schema_version': '1', 'experiment_id': 'fixture',
              'builds': [{'path': 'manifest.json', 'sha256': file_hash(build / 'manifest.json')}],
              'targets': [{'path': name, 'sha256': '0' * 64} for name in ('left.toml', 'right.toml')],
              'expected_requests': 128, 'timeout_seconds': 30}
    write_json(output / 'experiment.json', recipe, 'experiment')
    metadata = []
    for index, directory in enumerate((bad, good)):
        snapshot = output / f'run-{index}.json'
        write_json(snapshot, json.loads((directory / 'run.json').read_text()), 'run')
        metadata.append({'path': snapshot.name, 'sha256': file_hash(snapshot)})
    evaluate.receipt(output, 'complete.json', 'qualification', {
        'status': 'execution_complete', 'expected_requests': 128,
        'targets': [json.loads((p / 'target.json').read_text()) for p in (bad, good)],
        'experiment_sha256': file_hash(output / 'experiment.json'),
        'comparisons': [{'path': str(path), 'sha256': file_hash(path), 'build_id': rows[0]['build_id'],
                         'runs': metadata, 'repeats': 2,
                         'groups': [[r[k] for k in ('task_id', 'condition_id', 'mode')] for r in rows]}]})
    return output / 'complete.json', path


def test_offline_report_is_deterministic_and_keeps_unavailable(pipeline, tmp_path):
    completion, _ = completed(pipeline, tmp_path)
    first, second = tmp_path / 'report1', tmp_path / 'report2'
    report.report(completion, first)
    report.report(completion, second)
    for name in ('summary.json', 'report.rst'):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    summary = report.summarize(completion)
    row = summary['datasets'][0]['overall']
    assert summary['requests'] == 128
    assert row['numerical_groups']['unavailable'] > 0
    assert row['left']['checks_per_response']['missing_records']['fail'] > 0
    assert 'unavailable' in row['right']['checks_per_response']['missing_records']
    with pytest.raises(FileExistsError):
        report.report(completion, first)


@pytest.mark.parametrize('mutation', ['hash', 'count', 'inventory', 'groups', 'status'])
def test_report_refuses_incomplete_or_changed_evidence(pipeline, tmp_path, mutation):
    completion, path = completed(pipeline, tmp_path)
    data = json.loads(completion.read_text())
    if mutation == 'hash':
        path.write_bytes(path.read_bytes() + b'\n')
    elif mutation == 'count':
        data['data']['expected_requests'] += 1
    elif mutation == 'inventory':
        data['data']['comparisons'] = []
    elif mutation == 'groups':
        data['data']['comparisons'][0]['groups'].pop()
    else:
        data['data']['status'] = 'partial'
    completion.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report.summarize(completion)


@pytest.mark.parametrize('mutation', ['targets', 'runs', 'missing', 'hash', 'run_id',
                                    'build_id', 'manifest_sha256', 'source_commit',
                                    'binary_sha256', 'runtime_config'])
def test_report_binds_labels_to_run_identities(pipeline, tmp_path, mutation):
    completion, _ = completed(pipeline, tmp_path)
    receipt = json.loads(completion.read_text())
    data = receipt['data']
    item = data['comparisons'][0]
    if mutation == 'targets':
        data['targets'].reverse()
    elif mutation == 'runs':
        item['runs'].reverse()
    elif mutation == 'missing':
        del item['runs']
    else:
        reference = item['runs'][0]
        path = completion.parent / reference['path']
        meta = json.loads(path.read_text())
        if mutation in ('source_commit', 'binary_sha256'):
            meta['fingerprint'][mutation] = 'f' * len(meta['fingerprint'][mutation])
        elif mutation == 'runtime_config':
            meta['fingerprint']['runtime_config']['context_size'] = 1024
        elif mutation != 'hash':
            meta[mutation] = 'f' * 64 if mutation == 'manifest_sha256' else 'other'
        path.write_text(json.dumps(meta))
        if mutation != 'hash':
            reference['sha256'] = file_hash(path)
    completion.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        report.summarize(completion)


def test_report_bundle_relocates_without_model_or_server(pipeline, tmp_path):
    completion, comparison = completed(pipeline, tmp_path)
    value = json.loads(completion.read_text())
    value['data']['comparisons'][0]['path'] = os.path.relpath(comparison, completion.parent)
    completion.write_text(json.dumps(value))
    before = report.summarize(completion)
    relocated = tmp_path.parent / (tmp_path.name + '-relocated')
    shutil.copytree(tmp_path, relocated)
    (relocated / 'model').unlink()
    (relocated / 'binary').unlink()
    assert report.summarize(relocated / 'experiment/complete.json') == before


def test_prepare_admits_two_distinct_builds(recipe, monkeypatch, tmp_path):
    from adapters.util import fixture
    monkeypatch.setitem(fixture.TEXTS, 'ascii', 'XYZ')
    second = fixture.build(tmp_path / 'second')
    value = json.loads(recipe.read_text())
    value['builds'].append({'path': str(second / 'manifest.json'),
                            'sha256': file_hash(second / 'manifest.json')})
    value['expected_requests'] = 256
    recipe.write_text(json.dumps(value))
    monkeypatch.setattr(evaluate, 'Client', lambda _: pytest.fail('Offline admission contacted server'))
    admitted, builds, _, _ = evaluate.prepare(recipe)
    assert admitted['expected_requests'] == 256
    assert len({m['build_id'] for _, m, _ in builds}) == 2
    assert [str(b / 'manifest.json') for b, _, _ in builds] == [b['path'] for b in value['builds']]
    assert all(not (b / 'runs').exists() for b, _, _ in builds)


@pytest.fixture
def two_build_completion(tmp_path, monkeypatch):
    from adapters.util import fixture
    first = make_pipeline(tmp_path / 'first', monkeypatch)
    # A different task produces a distinct immutable build identity.
    monkeypatch.setitem(fixture.TEXTS, 'ascii', 'XYZ')
    second = make_pipeline(tmp_path / 'second', monkeypatch)
    builds = [(p[0], *load_build(p[0])) for p in (first, second)]
    targets = [json.loads((p / 'target.json').read_text()) for p in (first[2], first[1])]
    recipe = {'schema_version': '1', 'experiment_id': 'two-build-control',
              'builds': [{'path': str(b / 'manifest.json'), 'sha256': file_hash(b / 'manifest.json')}
                         for b, _, _ in builds],
              'targets': [{'path': name, 'sha256': '0' * 64} for name in ('left.toml', 'right.toml')],
              'expected_requests': 256, 'timeout_seconds': 30}
    calls = []
    @contextmanager
    def owned(target, *args):
        calls.append(('start', target['target_id']))
        try:
            yield
        finally:
            calls.append(('stop', target['target_id']))
    def saved_run(build, target):
        calls.append(('run', build, target['target_id']))
        return build / 'runs' / target['target_id']
    monkeypatch.setattr(evaluate, 'prepare', lambda _: (recipe, builds, targets, None))
    monkeypatch.setattr(evaluate, 'server', owned)
    monkeypatch.setattr(evaluate, 'Client', lambda _: nullcontext())
    monkeypatch.setattr(evaluate, 'run', saved_run)
    completion = evaluate.evaluate('unused', tmp_path / 'experiment')
    assert calls == [entry for target in targets for entry in
                     [('start', target['target_id']),
                      *[('run', b, target['target_id']) for b, _, _ in builds],
                      ('stop', target['target_id'])]]
    return completion, [m['build_id'] for _, m, _ in builds]


def test_two_build_report_totals_labels_and_portability(two_build_completion, tmp_path):
    completion, identities = two_build_completion
    summary = report.summarize(completion)
    assert len(set(identities)) == 2
    assert [d['build_id'] for d in summary['datasets']] == identities
    assert summary['requests'] == 256
    assert [t['target_id'] for t in summary['targets']] == ['mock-unpatched', 'mock-patched']
    for dataset in summary['datasets']:
        dataset = dataset['overall']
        assert dataset['left']['attempts'] == dataset['right']['attempts'] == 64
        assert dataset['left']['checks_per_response']['missing_records']['fail'] > 0
        assert not any(c.get('fail', 0) for c in dataset['right']['checks_per_response'].values())
    relocated = tmp_path.parent / (tmp_path.name + '-relocated')
    shutil.copytree(tmp_path, relocated)
    for lane in ('first', 'second'):
        for name in ('model', 'binary'):
            (relocated / lane / name).unlink()
    report.report(completion, tmp_path / 'report')
    report.report(relocated / 'experiment/complete.json', relocated / 'report')
    for name in ('summary.json', 'report.rst'):
        assert (tmp_path / 'report' / name).read_bytes() == (relocated / 'report' / name).read_bytes()


@pytest.mark.parametrize('mutation', ['order', 'cross_build_run', 'missing', 'duplicate'])
def test_two_build_report_rejects_mixed_inventory(two_build_completion, mutation):
    completion, _ = two_build_completion
    receipt = json.loads(completion.read_text())
    items = receipt['data']['comparisons']
    if mutation == 'order':
        items.reverse()
    elif mutation == 'cross_build_run':
        items[0]['runs'][0] = items[1]['runs'][0]
    elif mutation == 'missing':
        items.pop()
    else:
        items[1] = items[0]
    completion.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        report.summarize(completion)


def test_execution_composes_interfaces_and_retains_failure(tmp_path, monkeypatch):
    recipe = {'schema_version': '1', 'experiment_id': 'unit', 'builds': [{'path': 'manifest.json', 'sha256': '0'*64}],
              'targets': [{'path': n, 'sha256': '0'*64} for n in ('a', 'b')],
              'timeout_seconds': 30, 'expected_requests': 4}
    calls = []
    @contextmanager
    def owned(*args):
        calls.append('start')
        try:
            yield
        finally:
            calls.append('stop')
    def fail(*args):
        calls.append('run')
        raise RuntimeError('retained failure')
    monkeypatch.setattr(evaluate, 'prepare', lambda _: (recipe, [(tmp_path, {'conditions': []}, {})], [{'base_url': 'unused'}]*2, None))
    monkeypatch.setattr(evaluate, 'server', owned)
    monkeypatch.setattr(evaluate, 'Client', lambda _: nullcontext())
    monkeypatch.setattr(evaluate, 'run', fail)
    output = tmp_path / 'attempt'
    with pytest.raises(RuntimeError, match='retained failure'):
        evaluate.evaluate('ignored', output)
    assert calls == ['start', 'run', 'stop']
    assert (output / 'failure.json').exists()
    assert not (output / 'complete.json').exists()
