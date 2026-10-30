"""Compose existing MURPHY stages on two owned local servers. No cloud or retries."""
import argparse
from contextlib import contextmanager
from importlib.metadata import version
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from murphy.client import Client, now
from murphy.core.compare import compare
from murphy.core.grade import grade
from murphy.core.run import check_properties, run
from murphy.util.io import encoded, file_hash, load_build, read_json, validate, write_json
from murphy.util.vocabulary import Vocabulary


class ExperimentInterrupted(BaseException):
    """Must not be swallowed by transient HTTP readiness handling."""


def checked_file(base, item):
    path = (base / item['path']).resolve()
    if file_hash(path) != item['sha256']:
        raise ValueError(f'Input hash mismatch: {path}')
    return path


def prepare(recipe_path):
    """Offline admission only; paths in the recipe resolve relative to its file."""
    recipe_path = Path(recipe_path).resolve()
    recipe = read_json(recipe_path, 'experiment')
    builds = []
    for item in recipe['builds']:
        manifest_path = checked_file(recipe_path.parent, item)
        if manifest_path.name != 'manifest.json':
            raise ValueError('Build inputs must name manifest.json')
        manifest, tasks = load_build(manifest_path.parent)
        builds.append((manifest_path.parent, manifest, tasks))
    if len({p for p, _, _ in builds}) != len(builds) or len({m['build_id'] for _, m, _ in builds}) != len(builds):
        raise ValueError('Duplicate build')
    targets = []
    for item in recipe['targets']:
        path = checked_file(recipe_path.parent, item)
        with path.open('rb') as stream:
            target = validate(tomllib.load(stream), 'target')
        for kind in ('model', 'binary'):
            target[kind + '_path'] = str((path.parent / target[kind + '_path']).resolve())
            if file_hash(target[kind + '_path']) != target[kind + '_sha256']:
                raise ValueError(f'Target {kind} hash mismatch')
        runtime = target['runtime_config']
        required = {'context_size', 'batch_size', 'ubatch_size', 'gpu_layers', 'flash_attention'}
        if not required <= runtime.keys():
            raise ValueError('Owned execution requires explicit runtime sizes and GPU/flash settings')
        if (runtime['backend'] == 'cpu') != (runtime['gpu_layers'] == 0):
            raise ValueError('Backend and GPU layers disagree')
        port = int(target['base_url'].rsplit(':', 1)[1])
        if not 1024 <= port <= 65535:
            raise ValueError('Use an unprivileged local port')
        if target['reasoning_capable'] and 'template' not in recipe:
            raise ValueError('Reasoning requires an explicit hash-pinned template')
        targets.append(target)
    if targets[0]['target_id'] == targets[1]['target_id']:
        raise ValueError('Targets need distinct attempt IDs')
    for key in ('model_sha256', 'runtime_config', 'reasoning_capable'):
        if targets[0][key] != targets[1][key]:
            raise ValueError(f'Paired targets disagree: {key}')
    requests = 2 * sum(sum(len(c['task_ids']) for c in m['conditions']) * len(m['modes']) * m['repeats']
                       for _, m, _ in builds)
    if requests != recipe['expected_requests']:
        raise ValueError('Expected request count disagrees with manifests')
    for build, manifest, _ in builds:
        for target in targets:
            if (build / 'runs' / target['target_id']).exists():
                raise FileExistsError('Run already exists; choose new attempt IDs')
            if any(c['reasoning'] == 'on' for c in manifest['conditions']) and not target['reasoning_capable']:
                raise ValueError('Build requires reasoning capability')
        if (build / 'compare' / f"{targets[0]['target_id']}__{targets[1]['target_id']}.jsonl").exists():
            raise FileExistsError('Comparison already exists')
    template = checked_file(recipe_path.parent, recipe['template']) if 'template' in recipe else None
    Vocabulary(targets[0]['model_path'])
    return recipe, builds, targets, template


def receipt(output, name, kind, data):
    write_json(output / name, {'schema_version': '1', 'kind': kind, 'data': data}, 'receipt')


@contextmanager
def server(target, template, output):
    runtime = target['runtime_config']
    port = int(target['base_url'].rsplit(':', 1)[1])
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))  # Never attach to an existing listener.
    binary = Path(target['binary_path'])
    command = [str(binary), '--model', target['model_path'], '--host', '127.0.0.1', '--port', str(port),
               '--threads', '1', '--threads-batch', '1', '--parallel', '1', '--seed', '42', '--jinja',
               '--no-warmup', '--verbosity', '5', '--reasoning-format', 'deepseek',
               '--chat-template-kwargs', '{"enable_thinking":false}']
    for key, flag in [('context_size', '--ctx-size'), ('batch_size', '--batch-size'),
                      ('ubatch_size', '--ubatch-size'), ('gpu_layers', '--n-gpu-layers'),
                      ('flash_attention', '--flash-attn')]:
        command.extend([flag, str(runtime[key])])
    if template:
        command.extend(['--chat-template-file', str(template)])
    env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_')}
    env['LD_LIBRARY_PATH'] = str(binary.parent)
    if sys.platform == 'darwin':
        env['DYLD_LIBRARY_PATH'] = str(binary.parent)
    name = target['target_id']
    with (output / (name + '-server.log')).open('xb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            receipt(output, name + '-launch.json', 'launch', {
                'at': now(), 'pid': process.pid, 'command': command, 'target': target,
                'libraries': {p.name: file_hash(p) for p in binary.parent.iterdir()
                              if p.is_file() and ('.so' in p.name or p.suffix == '.dylib')}})
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Owned server exited; see preserved log')
                try:
                    with Client(target['base_url'], timeout=1) as client:
                        props = client.properties()
                    break
                except Exception:
                    time.sleep(0.2)
            else:
                raise TimeoutError('Server readiness deadline')
            check_properties(target, props)
            if runtime['backend'] == 'cuda' and 'assigned to device CUDA' not in (output / (name + '-server.log')).read_text(errors='replace'):
                raise ValueError('CUDA target has no positive layer-placement evidence in the server log')
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                except ExperimentInterrupted:
                    process.kill()
                    process.wait(timeout=10)
                    raise
            receipt(output, name + '-shutdown.json', 'shutdown', {
                'at': now(), 'pid': process.pid, 'returncode': process.returncode, 'owned_child_stopped': True})


def evaluate(recipe_path, output):
    recipe, builds, targets, template = prepare(recipe_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'experiment.json', recipe, 'experiment')
    source_root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(source_root)): file_hash(p)
               for folder in ('murphy', 'adapters', 'scripts') for p in (source_root / folder).rglob('*')
               if p.is_file() and p.suffix in ('.py', '.json', '.txt')}
    receipt(output, 'source.json', 'qualification', {
        'files': sources, 'python': sys.version, 'platform': platform.platform(),
        'requirements_sha256': file_hash(source_root / 'requirements.txt'),
        'packages': {name: version(name) for name in ('httpx', 'jsonschema', 'tokenizers', 'jinja2', 'gguf')}})
    started = time.monotonic()
    def interrupt(signum, frame):
        raise ExperimentInterrupted(f'Experiment interrupted by signal {signum}; no replay')
    signals = (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)
    previous = {s: signal.signal(s, interrupt) for s in signals}
    signal.alarm(recipe['timeout_seconds'])
    try:
        runs = {str(build): [] for build, _, _ in builds}
        for target in targets:
            with server(target, template, output):
                for build, manifest, tasks in builds:
                    with Client(target['base_url']) as client:
                        for condition in manifest['conditions']:
                            if 'prompt_wrapper' in condition:
                                wrapper = condition['prompt_wrapper']
                                for task_id in condition['task_ids']:
                                    prompt = tasks[task_id]['prompt']
                                    if client.apply_template(prompt) != wrapper['prefix'] + prompt + wrapper['suffix']:
                                        raise ValueError('Native template disagrees with pinned prompt wrapper')
                    directory = run(build, target)
                    runs[str(build)].append(directory)
                    grade(build, directory)
        comparisons = []
        for build, manifest, _ in builds:
            path = compare(build, *runs[str(build)])
            manifest_snapshot = output / 'runs' / manifest['build_id'] / 'manifest.json'
            manifest_snapshot.parent.mkdir(parents=True, exist_ok=True)
            with manifest_snapshot.open('xb') as handle:
                handle.write((build / 'manifest.json').read_bytes())
            metadata = []
            for directory in runs[str(build)]:
                meta = read_json(directory / 'run.json', 'run')
                snapshot = output / 'runs' / manifest['build_id'] / (meta['target_id'] + '.json')
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                write_json(snapshot, meta, 'run')
                metadata.append({'path': os.path.relpath(snapshot, output), 'sha256': file_hash(snapshot)})
            comparisons.append({'path': os.path.relpath(path, output), 'sha256': file_hash(path),
                                'manifest': {'path': os.path.relpath(manifest_snapshot, output),
                                             'sha256': file_hash(manifest_snapshot)},
                                'runs': metadata,
                                'build_id': manifest['build_id'], 'repeats': manifest['repeats'],
                                'groups': [[task, c['id'], mode] for c in manifest['conditions']
                                           for task in c['task_ids'] for mode in manifest['modes']]})
        receipt(output, 'complete.json', 'qualification', {
            'status': 'execution_complete', 'experiment_sha256': file_hash(output / 'experiment.json'),
            'source_receipt_sha256': file_hash(output / 'source.json'),
            'targets': targets,
            'expected_requests': recipe['expected_requests'], 'elapsed_seconds': time.monotonic() - started,
            'comparisons': comparisons, 'limits': 'Execution complete does not mean all checks passed.'})
    except BaseException as exc:
        receipt(output, 'failure.json', 'failure', {'at': now(), 'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        signal.alarm(0)
        for s, handler in previous.items():
            signal.signal(s, handler)
    return output / 'complete.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', type=Path)
    parser.add_argument('--check', action='store_true', help='Offline admission; no output or server')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.check:
        recipe, _, _, _ = prepare(args.recipe)
        print(encoded({'admitted': True, 'requests': recipe['expected_requests']}).decode(), end='')
    else:
        if args.output is None:
            parser.error('--output is required for execution')
        print(evaluate(args.recipe, args.output))


if __name__ == '__main__':
    main()
