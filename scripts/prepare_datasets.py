"""Partition an admitted mixed build into immutable dataset-local builds; no inference."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from murphy.util.io import JsonlWriter, digest, file_hash, load_build, write_json

DATASETS = {'flores': 'flores_plus', 'belebele': 'belebele', 'emoji': 'unicode_emoji'}


def partition(parent, output):
    parent, output = Path(parent), Path(output)
    manifest, tasks = load_build(parent)
    groups = {name: [] for name in DATASETS.values()}
    for task in tasks.values():
        prefix = task['task_id'].split('-')[0]
        if prefix not in DATASETS:
            raise ValueError('Unknown source task prefix; refusing partial partition')
        groups[DATASETS[prefix]].append(task)
    planned = []
    for dataset, rows in groups.items():
        if not rows:
            raise ValueError('All three datasets must be present')
        child = deepcopy(manifest)
        child['dataset'] = dataset
        child['task_count'] = len(rows)
        child['source']['partition'] = {
            'parent_build_id': manifest['build_id'],
            'parent_manifest_sha256': file_hash(parent / 'manifest.json'),
            'parent_tasks_sha256': manifest['tasks_sha256'],
            'dataset': dataset,
            'note': 'Source metadata is the immutable parent recipe/provenance; this build contains only the selected dataset.'}
        child['source']['allocation'] = {dataset: len(rows)}
        ids = {t['task_id'] for t in rows}
        child['conditions'] = [{**c, 'task_ids': [i for i in c['task_ids'] if i in ids]}
                               for c in child['conditions'] if ids.intersection(c['task_ids'])]
        if not child['conditions']:
            raise ValueError('Dataset has no request conditions')
        child['build_id'] = dataset + '-' + digest({'manifest': child, 'tasks': rows})[:12]
        destination = output / dataset / child['build_id']
        if destination.exists():
            raise FileExistsError(destination)
        planned.append((destination, child, rows))
    # Preflight every destination before creating any build. Never move historical runs.
    for destination, child, rows in planned:
        destination.mkdir(parents=True, exist_ok=False)
        with JsonlWriter(destination / 'tasks.jsonl', 'task') as writer:
            for row in rows:
                writer.append(row)
        child['tasks_sha256'] = file_hash(destination / 'tasks.jsonl')
        write_json(destination / 'manifest.json', child, 'manifest')
        load_build(destination)
    return [p for p, _, _ in planned]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data'))
    args = parser.parse_args()
    for path in partition(args.build, args.output):
        print(path)
