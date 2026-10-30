import json
import shutil
from contextlib import contextmanager, nullcontext
import pytest
from test_experiment import recipe
from test_pipeline import make_pipeline
from murphy.util.io import file_hash, load_build
from scripts import evaluate, report

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
