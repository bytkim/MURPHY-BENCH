"""Endpoint presentation controls using saved mock responses, never inference."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from test_experiment import completed, two_build_completion
from test_pipeline import make_pipeline
from murphy.util.io import file_hash, read_jsonl
from scripts import report


@pytest.fixture
def saved(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path, monkeypatch, chat_usage=True)
    completion, comparison = completed(pipeline, tmp_path)
    return completion, comparison, pipeline[0]


def modes(dataset):
    return {(endpoint['endpoint'], mode['mode']): mode
            for endpoint in dataset['endpoints'] for mode in endpoint['modes']}


def test_all_five_measurements_are_partitioned_and_reconcile(saved):
    completion, comparison, _ = saved
    summary = report.summarize(completion)
    dataset = summary['datasets'][0]
    rows = read_jsonl(comparison, 'comparison.v2')
    assert summary['schema_version'] == '3'
    assert summary['primary_grouping'] == ['dataset', 'endpoint', 'mode', 'target']
    partitioned = modes(dataset)
    assert set(partitioned) == {(e, m) for e in ('/completion', '/v1/chat/completions')
                                for m in ('nonstream', 'stream')}
    for (endpoint, mode_name), mode in partitioned.items():
        selected = [r for r in rows if r['condition_id'] in mode['condition_ids'] and r['mode'] == mode_name]
        assert mode['numerical_agreement'] == report.fractions([r['measurements']['numerical_agreement'] for r in selected])
        for side in ('left', 'right'):
            for name, metric in mode[side]['measurements'].items():
                assert metric == report.fractions([r['measurements'][name][side] for r in selected])
            assert mode[side]['attempts'] == sum(r[side]['attempts'] for r in selected)
            assert mode[side]['target_id'] == summary['targets'][side == 'right']['target_id']
    for side in ('left', 'right'):
        assert sum(m[side]['attempts'] for m in partitioned.values()) == dataset['overall'][side]['attempts']
        assert sum(m[side]['complete'] for m in partitioned.values()) == dataset['overall'][side]['complete']
        for name, total in dataset['overall'][side]['measurements'].items():
            assert sum(m[side]['measurements'][name]['numerator'] for m in partitioned.values()) == total['numerator']
            assert sum(m[side]['measurements'][name]['denominator'] or 0 for m in partitioned.values()) == (total['denominator'] or 0)
        for grade, total in dataset['overall'][side]['task_grades'].items():
            assert sum(m[side]['task_grades'][grade] for m in partitioned.values()) == total


def test_chat_never_inherits_native_success_and_counts_remain_separate(saved, tmp_path):
    completion, _, _ = saved
    dataset = report.summarize(completion)['datasets'][0]
    for (endpoint, _), mode in modes(dataset).items():
        if endpoint == '/completion':
            assert mode['right']['measurements']['token_level_coverage']['value'] == 1
            continue
        assert mode['numerical_agreement']['value'] is None
        for side in ('left', 'right'):
            for name in ('token_level_coverage', 'actual_output_exposure'):
                metric = mode[side]['measurements'][name]
                assert metric['value'] is None and metric['denominator'] is None
                assert metric['unavailable_reasons'] == {'native_token_reference_unavailable': mode['groups']}
            assert mode[side]['measurements']['complete_response_rate']['value'] == 1
            checks = [g for g in mode['reference_breakdown'] if g['side'] == side and g['category'] == 'missing_records']
            assert checks and all(g['reference'] == 'chat_usage_completion_tokens' for g in checks)
            shortfall = sum(g['record_accounting']['net_shortfall_sum'] for g in checks)
            assert shortfall > 0 if side == 'left' else shortfall == 0
    out = report.report(completion, tmp_path / 'rendered')
    text = (out / 'report.rst').read_text()
    assert text.index('Chat usage checks reconcile counts') < text.index('/completion / nonstream')
    chat = text.split('/v1/chat/completions / nonstream', 1)[1].split('Overall totals', 1)[0]
    assert 'token_level_coverage: unavailable' in chat
    assert 'actual_output_exposure: unavailable' in chat
    assert 'Paired numerical agreement: unavailable' in chat
    assert 'token_level_coverage: 0/' not in chat
    assert '100.00%' not in '\n'.join(line for line in chat.splitlines() if 'token_level_coverage:' in line)
    assert text.index('Overall totals') > text.index('/v1/chat/completions / stream')
    assert 'Instruction-following grades:' in text and 'Record integrity checks (separate from instruction following):' in text


def test_endpoint_names_come_from_verified_manifest_not_condition_names(saved):
    completion, comparison, build = saved
    receipt = json.loads(completion.read_text())
    item = receipt['data']['comparisons'][0]
    manifest_path = build / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    names = {c['id']: f'opaque-{i}' for i, c in enumerate(manifest['conditions'])}
    for c in manifest['conditions']:
        c['id'] = names[c['id']]
    manifest_path.write_text(json.dumps(manifest))
    rows = read_jsonl(comparison, 'comparison.v2')
    for r in rows:
        r['condition_id'] = names[r['condition_id']]
        for side in ('left', 'right'):
            for grade in r[side]['grades']:
                grade['condition_id'] = r['condition_id']
    comparison.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    item['sha256'] = file_hash(comparison)
    item['groups'] = [[r['task_id'], r['condition_id'], r['mode']] for r in rows]
    recipe_path = completion.parent / 'experiment.json'
    recipe = json.loads(recipe_path.read_text())
    recipe['builds'][0]['sha256'] = file_hash(manifest_path)
    recipe_path.write_text(json.dumps(recipe))
    receipt['data']['experiment_sha256'] = file_hash(recipe_path)
    for reference in item['runs']:
        p = completion.parent / reference['path']
        run = json.loads(p.read_text())
        run['manifest_sha256'] = file_hash(manifest_path)
        p.write_text(json.dumps(run))
        reference['sha256'] = file_hash(p)
    completion.write_text(json.dumps(receipt))
    result = report.summarize(completion)
    assert set(e['endpoint'] for e in result['datasets'][0]['endpoints']) == {'/completion', '/v1/chat/completions'}
    assert all(c.startswith('opaque-') for m in modes(result['datasets'][0]).values() for c in m['condition_ids'])


@pytest.mark.parametrize('mutation', ['bytes', 'ref_hash', 'wrong_build', 'missing', 'repeat', 'matrix'])
def test_snapshot_mismatch_rejected_even_with_original_available(two_build_completion, mutation):
    completion, _ = two_build_completion
    receipt = json.loads(completion.read_text())
    item = receipt['data']['comparisons'][0]
    snapshot = completion.parent / item['manifest']['path']
    if mutation == 'bytes':
        snapshot.write_bytes(snapshot.read_bytes() + b' ')
    elif mutation == 'ref_hash':
        item['manifest']['sha256'] = '0' * 64
    elif mutation == 'wrong_build':
        item['manifest'] = receipt['data']['comparisons'][1]['manifest']
    elif mutation == 'missing':
        snapshot.unlink()
    elif mutation == 'repeat':
        item['repeats'] += 1
    else:
        item['groups'].pop()
    completion.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, FileNotFoundError)):
        report.summarize(completion)


def test_explicit_snapshots_are_exact_and_portable_without_original_manifests(two_build_completion, tmp_path):
    completion, _ = two_build_completion
    receipt = json.loads(completion.read_text())
    recipe = json.loads((completion.parent / 'experiment.json').read_text())
    before = report.summarize(completion)
    for item, build in zip(receipt['data']['comparisons'], recipe['builds']):
        assert (completion.parent / item['manifest']['path']).read_bytes() == Path(build['path']).read_bytes()
    relocated = tmp_path.parent / (tmp_path.name + '-snapshot-relocated')
    shutil.copytree(tmp_path, relocated)
    for build in recipe['builds']:
        Path(build['path']).unlink()
        (relocated / Path(build['path']).relative_to(tmp_path)).unlink()
    for lane in ('first', 'second'):
        for name in ('model', 'binary'):
            (relocated / lane / name).unlink()
    assert report.summarize(relocated / 'experiment/complete.json') == before


def test_report_does_not_modify_saved_evidence(saved, tmp_path):
    completion, comparison, build = saved
    paths = [completion, comparison, build / 'manifest.json', build / 'tasks.jsonl']
    before = {p: file_hash(p) for p in paths}
    report.report(completion, tmp_path / 'readonly-report')
    assert before == {p: file_hash(p) for p in paths}


def test_legacy_report_requires_manifest_instead_of_guessing_endpoint(saved):
    completion, _, build = saved
    (build / 'manifest.json').unlink()
    with pytest.raises(FileNotFoundError):
        report.summarize(completion)


def test_accounting_shortfalls_and_excesses_do_not_cancel():
    def row(delta):
        check = {'category': 'missing_records', 'reference': 'chat_usage_completion_tokens',
                 'status': 'fail', 'details': {'expected_records': 10, 'observed_records': 10 + delta,
                                             'count_delta': delta}}
        return {'condition_id': 'chat', 'mode': 'stream',
                'left': {'checks_by_repeat': [[check]]}, 'right': {'checks_by_repeat': [[deepcopy(check)]]}}
    for group in report.reference_breakdown([row(-3), row(3)]):
        counts = group['record_accounting']
        assert counts['expected_records'] == counts['observed_records'] == 20
        assert counts['net_shortfall_sum'] == counts['net_excess_sum'] == 3
        assert counts['equal_count_responses'] == 0
