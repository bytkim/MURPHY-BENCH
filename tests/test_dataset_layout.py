import pytest

from adapters.util.fixture import build
from murphy.util.io import JsonlWriter, file_hash, load_build, write_json
from scripts.prepare_datasets import partition


def parent(tmp_path):
    manifest, tasks = load_build(build(tmp_path))
    selected = list(tasks.values())[:3]
    for prefix, task in zip(('flores', 'belebele', 'emoji'), selected):
        task['task_id'] = prefix + '-control'
    manifest.update(dataset='mixed', build_id='mixed-control', task_count=3)
    manifest['conditions'] = [{**manifest['conditions'][0], 'task_ids': [t['task_id'] for t in selected]}]
    path = tmp_path / 'parent'
    path.mkdir()
    with JsonlWriter(path / 'tasks.jsonl', 'task') as writer:
        for task in selected:
            writer.append(task)
    manifest['tasks_sha256'] = file_hash(path / 'tasks.jsonl')
    write_json(path / 'manifest.json', manifest, 'manifest')
    return path


def test_partition_is_exact_reproducible_and_exclusive(tmp_path):
    source = parent(tmp_path)
    original = {p.name: file_hash(p) for p in source.iterdir()}
    paths = partition(source, tmp_path / 'data')
    again = partition(source, tmp_path / 'repeat')
    _, old = load_build(source)
    combined = {}
    for path, repeated in zip(paths, again):
        manifest, tasks = load_build(path)
        assert path.parent.name == manifest['dataset']
        assert manifest['source']['partition']['parent_manifest_sha256'] == original['manifest.json']
        assert manifest['task_count'] == 1
        assert manifest['conditions'][0]['task_ids'] == list(tasks)
        assert not (path / 'runs').exists()  # No fabricated results.
        combined.update(tasks)
        for name in ('manifest.json', 'tasks.jsonl'):
            assert (path / name).read_bytes() == (repeated / name).read_bytes()
    assert combined == old
    with pytest.raises(FileExistsError):
        partition(source, tmp_path / 'data')
    assert {p.name: file_hash(p) for p in source.iterdir()} == original


def test_all_destinations_preflight_before_writing(tmp_path):
    source = parent(tmp_path)
    planned = partition(source, tmp_path / 'first')
    output = tmp_path / 'blocked'
    (output / planned[-1].parent.name / planned[-1].name).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        partition(source, output)
    assert not (output / planned[0].parent.name).exists()


@pytest.mark.parametrize('case', ['unknown_source', 'missing_source', 'missing_conditions'])
def test_partition_rejects_incomplete_inputs_before_writing(tmp_path, monkeypatch, case):
    source = parent(tmp_path)
    manifest, tasks = load_build(source)
    if case == 'unknown_source':
        tasks['emoji-control']['task_id'] = 'unknown-control'
    elif case == 'missing_source':
        del tasks['emoji-control']
    else:
        manifest['conditions'][0]['task_ids'].remove('emoji-control')
    # Exercise partition admission separately from the already-tested build reader.
    monkeypatch.setattr('scripts.prepare_datasets.load_build', lambda _: (manifest, tasks))
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError):
        partition(source, output)
    assert not output.exists()
