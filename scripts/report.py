"""Endpoint-separated summary v3 from unchanged v2 comparisons.

Primary view: dataset, endpoint, mode, ordered target. Overall totals are
secondary. Offline only: no inference, regrading or repair of saved evidence.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from murphy.util.io import encoded, file_hash, read_json, read_jsonl, validate


def fractions(values):
    """Sum declared counts, never average rates or treat unavailable as zero."""
    statuses = dict(Counter(v['status'] for v in values))
    known = [v for v in values if v['denominator'] is not None and v['denominator'] > 0]
    numerator = sum(v['numerator'] for v in known)
    denominator = sum(v['denominator'] for v in known)
    return {'numerator': numerator, 'denominator': denominator or None,
            'value': numerator / denominator if denominator else None,
            'group_statuses': statuses,
            'unavailable_reasons': dict(Counter(v.get('details', {}).get('reason', 'unspecified')
                                              for v in values if v['status'] == 'unavailable')),
            'scope': 'available denominators only; see group statuses'}


def reference_breakdown(rows):
    """Keep whole-response accounting separate from positional checks.

    Count shortfalls/excesses are summed per response, never cancelled against
    one another or relabelled as a positional coverage fraction.
    """
    groups = {}
    for row in rows:
        for side in ('left', 'right'):
            for repeat in row[side]['checks_by_repeat']:
                for check in repeat:
                    d = check['details']
                    key = (row['condition_id'], row['mode'], side, check['category'], check['reference'])
                    group = groups.setdefault(key, {'condition_id': key[0], 'mode': key[1], 'side': side,
                        'category': key[3], 'reference': key[4], 'statuses': Counter(), 'unavailable_reasons': Counter()})
                    group['statuses'][check['status']] += 1
                    if check['status'] == 'unavailable':
                        group['unavailable_reasons'][d.get('reason', 'unspecified')] += 1
                    if check['reference'] == 'chat_usage_completion_tokens':
                        counts = group.setdefault('record_accounting', {
                            'scope': 'whole_response', 'eligible_responses': 0, 'equal_count_responses': 0,
                            'expected_records': 0, 'observed_records': 0, 'net_shortfall_sum': 0, 'net_excess_sum': 0})
                        if check['status'] != 'unavailable':
                            expected, observed, delta = (d[k] for k in ('expected_records', 'observed_records', 'count_delta'))
                            if (any(type(x) is not int for x in (expected, observed, delta))
                                    or min(expected, observed) < 0 or delta != observed - expected
                                    or check['status'] != ('pass' if delta == 0 else 'fail')):
                                raise ValueError('Invalid chat record accounting')
                            counts['eligible_responses'] += 1
                            counts['equal_count_responses'] += delta == 0
                            counts['expected_records'] += expected
                            counts['observed_records'] += observed
                            counts['net_shortfall_sum'] += max(-delta, 0)
                            counts['net_excess_sum'] += max(delta, 0)
    return [groups[k] for k in sorted(groups)]


def summarize_rows(rows, targets):
    """Use the same arithmetic at every level; never turn unavailable into zero."""
    result = {'groups': len(rows), 'reference_breakdown': reference_breakdown(rows),
              'numerical_groups': dict(Counter(r['numerical']['status'] for r in rows)),
              'numerical_agreement': fractions([r['measurements']['numerical_agreement'] for r in rows])}
    for side, target in zip(('left', 'right'), targets):
        checks = {}
        for row in rows:
            for repeat in row[side]['checks_by_repeat']:
                for check in repeat:
                    checks.setdefault(check['category'], Counter())[check['status']] += 1
        result[side] = {
            'target_id': target['target_id'],
            'attempts': sum(r[side]['attempts'] for r in rows),
            'complete': sum(r[side]['complete_responses'] for r in rows),
            'unstable_groups': sum(not r[side]['stable_repeats'] for r in rows),
            'task_grades': {name: sum(g['pass'] is value for r in rows for g in r[side]['grades'])
                            for name, value in [('pass', True), ('fail', False), ('unavailable', None)]},
            'checks_per_response': {k: dict(v) for k, v in sorted(checks.items())},
            'measurements': {name: fractions([r['measurements'][name][side] for r in rows])
                             for name in ('token_level_coverage', 'complete_response_rate',
                                          'actual_output_exposure', 'task_compliance')}}
    return result


def summarize(completion):
    completion = Path(completion).resolve()
    receipt = read_json(completion, 'receipt')
    data = receipt['data']
    if receipt['kind'] != 'qualification' or data.get('status') != 'execution_complete':
        raise ValueError('A completed experiment receipt is required')
    recipe_path = completion.parent / 'experiment.json'
    recipe = read_json(recipe_path, 'experiment')
    if file_hash(recipe_path) != data['experiment_sha256']:
        raise ValueError('Experiment identity mismatch')
    if len(data['comparisons']) != len(recipe['builds']):
        raise ValueError('Incomplete comparison inventory')
    if len(data['targets']) != 2:
        raise ValueError('Expected two target identities')
    targets = [validate(target, 'target') for target in data['targets']]
    if targets[0]['target_id'] == targets[1]['target_id'] or any(
            targets[0][key] != targets[1][key] for key in ('model_sha256', 'runtime_config')):
        raise ValueError('Paired report target identities disagree')
    datasets, identities, attempts = [], set(), 0
    for item, build in zip(data['comparisons'], recipe['builds']):
        if len(item.get('runs', [])) != 2:
            raise ValueError('Two hash-bound run identities are required')
        runs = []
        for reference, target in zip(item['runs'], targets):
            run_path = (completion.parent / reference['path']).resolve()
            if file_hash(run_path) != reference['sha256']:
                raise ValueError('Run metadata hash mismatch')
            meta = read_json(run_path, 'run')
            if (meta['build_id'] != item['build_id'] or meta['manifest_sha256'] != build['sha256']
                    or meta['target_id'] != target['target_id'] or any(
                        meta['fingerprint'][key] != target[key] for key in
                        ('model_sha256', 'binary_sha256', 'source_commit', 'runtime_config'))):
                raise ValueError('Run identity disagrees with report target or build')
            runs.append(meta)
        path = (completion.parent / item['path']).resolve()
        if file_hash(path) != item['sha256']:
            raise ValueError('Comparison hash mismatch')
        # New bundles carry an explicit snapshot. Legacy bundles retain the
        # manifest beside their canonical build/compare/ output; never infer an
        # endpoint from a condition name or from the availability of a metric.
        reference = item.get('manifest')
        manifest_path = (completion.parent / reference['path'] if reference is not None
                         else path.parent.parent / 'manifest.json')
        if ((reference is not None and reference['sha256'] != build['sha256'])
                or file_hash(manifest_path) != build['sha256']):
            raise ValueError('Report manifest hash mismatch')
        manifest = read_json(manifest_path, 'manifest')
        conditions = {c['id']: c for c in manifest['conditions']}
        if (manifest['build_id'] != item['build_id'] or manifest['repeats'] != item['repeats']
                or len(conditions) != len(manifest['conditions'])):
            raise ValueError('Report manifest identity mismatch')
        rows = read_jsonl(path, 'comparison.v2')
        if not rows:
            raise ValueError('Empty comparison')
        if any(r['references']['model_sha256'] != targets[0]['model_sha256'] for r in rows):
            raise ValueError('Comparison model disagrees with target identities')
        group_ids = {(r['task_id'], r['condition_id'], r['mode']) for r in rows}
        planned_groups = {(task, c['id'], mode) for c in manifest['conditions']
                          for task in c['task_ids'] for mode in manifest['modes']}
        if group_ids != planned_groups or group_ids != {tuple(g) for g in item['groups']} or any(
                r['build_id'] != item['build_id'] or any(
                    r[s]['attempts'] != item['repeats'] or len(r[s]['grades']) != item['repeats']
                    or len(r[s]['checks_by_repeat']) != item['repeats'] for s in ('left', 'right')) for r in rows):
            raise ValueError('Comparison matrix disagrees with completion receipt')
        identity = {(r['build_id'], r['left_run_id'], r['right_run_id']) for r in rows}
        if identity != {(item['build_id'], runs[0]['run_id'], runs[1]['run_id'])}:
            raise ValueError('Comparison sides disagree with ordered run identities')
        if len(group_ids) != len(rows) or len(identity) != 1 or next(iter(identity)) in identities:
            raise ValueError('Duplicate or mixed comparison identities')
        identities.update(identity)
        result = {'dataset': manifest['dataset'], 'build_id': manifest['build_id'],
                  'manifest_sha256': build['sha256'], 'comparison_sha256': item['sha256'],
                  'endpoints': [], 'overall': summarize_rows(rows, targets)}
        for endpoint in dict.fromkeys(c['endpoint'] for c in manifest['conditions']):
            modes = []
            for mode in manifest['modes']:
                selected = [r for r in rows if r['mode'] == mode
                            and conditions[r['condition_id']]['endpoint'] == endpoint]
                modes.append({'mode': mode, 'condition_ids': sorted({r['condition_id'] for r in selected}),
                              **summarize_rows(selected, targets)})
            result['endpoints'].append({'endpoint': endpoint, 'modes': modes})
        attempts += sum(result['overall'][side]['attempts'] for side in ('left', 'right'))
        datasets.append(result)
    if attempts != recipe['expected_requests'] or attempts != data['expected_requests']:
        raise ValueError('Comparison attempts disagree with experiment')
    return {'schema_version': '3', 'experiment_id': recipe['experiment_id'],
            'primary_grouping': ['dataset', 'endpoint', 'mode', 'target'],
            'targets': [{k: target[k] for k in ('target_id', 'source_commit', 'model_sha256', 'binary_sha256', 'runtime_config')}
                        for target in targets],
            'completion_sha256': file_hash(completion), 'requests': attempts, 'datasets': datasets,
            'limits': ['Repeats are not independent questions.', 'Failure categories overlap.',
                       'Unavailable checks are not passes.', 'Numerical agreement is not numerical accuracy.',
                       'Chat usage checks reconcile counts, not token identities or positional coverage.',
                       'Chat token identity, positional coverage, actual-output exposure and numerical agreement remain unavailable under current eligibility rules.',
                       'Comparisons are between builds within the same endpoint and mode, not numerical parity between endpoints.',
                       'Dataset overall metrics combine conditions and modes using available denominators only; they are secondary, not endpoint coverage claims.',
                       'This report summarizes saved comparisons; it does not independently regrade raw responses.']}


def metric_text(metric):
    value = ('unavailable' if metric['value'] is None else
             f"{metric['numerator']}/{metric['denominator']} ({metric['value']:.2%})")
    return (f"{value}; group statuses {metric['group_statuses']}; "
            f"unavailable reasons {metric['unavailable_reasons']}")


def report(completion, output):
    summary = summarize(completion)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_bytes(encoded(summary))
    lines = [summary['experiment_id'], '=' * len(summary['experiment_id']), '',
             f"Requests: {summary['requests']}. Left/right follow recipe target order.", '',
             *summary['limits'], '']
    for side, target in zip(('left', 'right'), summary['targets']):
        lines.extend([f"{side}: {target['target_id']}; source {target['source_commit']}",
                      f"Model SHA256: {target['model_sha256']}", f"Binary SHA256: {target['binary_sha256']}",
                      f"Runtime: {target['runtime_config']}", ''])
    for dataset in summary['datasets']:
        title = f"{dataset['dataset']} / {dataset['build_id']}"
        lines.extend([title, '-' * len(title), ''])
        for endpoint in dataset['endpoints']:
            for mode in endpoint['modes']:
                title = f"{endpoint['endpoint']} / {mode['mode']}"
                lines.extend([title, '~' * len(title), '', f"Conditions: {mode['condition_ids']}", ''])
                for side in ('left', 'right'):
                    row = mode[side]
                    lines.extend([f"{side}: {row['target_id']}", '',
                                  f"Responses: {row['complete']}/{row['attempts']} complete; {row['unstable_groups']} unstable groups.",
                                  f"Instruction-following grades: {row['task_grades']}", ''])
                    for name, metric in row['measurements'].items():
                        lines.extend([f"* {name}: {metric_text(metric)}", ''])
                    lines.extend(['Record integrity checks (separate from instruction following):', ''])
                    for group in mode['reference_breakdown']:
                        if group['side'] != side:
                            continue
                        lines.extend([f"* {group['condition_id']} / {group['category']}",
                                      f"  Reference: {group['reference']}; statuses: {dict(group['statuses'])}."])
                        if group['unavailable_reasons']:
                            lines.append(f"  Unavailable: {dict(group['unavailable_reasons'])}.")
                        if 'record_accounting' in group:
                            c = group['record_accounting']
                            lines.extend([f"  Equal counts: {c['equal_count_responses']}/{c['eligible_responses']} eligible responses.",
                                          f"  Records: {c['observed_records']} observed / {c['expected_records']} expected; "
                                          f"sum of response net shortfalls: {c['net_shortfall_sum']}; excesses: {c['net_excess_sum']}."])
                        lines.append('')
                lines.extend([f"Paired numerical agreement: {metric_text(mode['numerical_agreement'])}", '',
                              f"Numerical check statuses: {mode['numerical_groups']}", ''])
        lines.extend(['Overall totals (secondary; all endpoints and modes)', '',
                      f"Comparison groups: {dataset['overall']['groups']}", ''])
        for side in ('left', 'right'):
            row = dataset['overall'][side]
            lines.extend([f"{row['target_id']}: {row['complete']}/{row['attempts']} complete; task grades {row['task_grades']}", ''])
    (output / 'report.rst').write_text('\n'.join(lines) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('completion', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(report(args.completion, args.output))
