import json
import os
from pathlib import Path

import pytest
from jsonschema import ValidationError

from adapters.util.tokenizer_probe import write_probe
from murphy.util.io import CONTRACTS, contract_identity, encoded, validate, validator


def test_only_versioned_contracts_in_own_folders():
    assert CONTRACTS == Path(os.environ['MURPHY_SOURCE']) / 'murphy/contract'
    for path in CONTRACTS.rglob('*.schema.json'):
        assert path.name.startswith(path.parent.name + '-v')
    assert not (CONTRACTS / 'benchmark-plan').exists()
    assert not (CONTRACTS / 'tokenizer-probe').exists()
    assert sorted(p.name for p in (CONTRACTS / 'util/comparison').glob('*.json')) == [
        'comparison-v1.schema.json', 'comparison-v2.schema.json']


def test_receipt_alias_and_explicit_version():
    receipt = {'schema_version': '1', 'kind': 'qualification', 'data': {}}
    for name in ('receipt', 'receipt.v1', 'util/receipt', 'util/receipt.v1'):
        assert validate(receipt, name) == receipt
    with pytest.raises(FileNotFoundError):
        validator('util/receipt.v2')


@pytest.mark.parametrize('name', ['grade', 'manifest', 'policy', 'response', 'run', 'target', 'task'])
def test_core_contract_location_and_compatible_names(name):
    path = CONTRACTS / 'core' / name / f'{name}-v1.schema.json'
    assert path.is_file()
    assert not (CONTRACTS / name / f'{name}-v1.schema.json').exists()
    assert contract_identity(name) == ('core/' + name, None)
    assert validator(name).schema == validator('core/' + name + '.v1').schema


def test_probe_adapter_preserves_bytes_and_exclusive_output(tmp_path):
    records = [{'source': 'flores_plus', 'group': 'test', 'text': 'A€', 'hf_ids': [1, 2]}]
    path = tmp_path / 'probe.json'
    write_probe(path, records)
    assert path.read_bytes() == encoded({'schema_version': '1', 'records': records})
    with pytest.raises(FileExistsError):
        write_probe(path, records)
    invalid = tmp_path / 'invalid.json'
    with pytest.raises(ValidationError):
        write_probe(invalid, [{**records[0], 'hf_ids': [-1]}])
    assert not invalid.exists()
