"""Public setup is an explicit policy selection, never an implicit latest recipe."""
import os
from pathlib import Path
import subprocess
import sys

from adapters.core.policy import DEFAULT, load


def test_public_preparation_invocation_is_explicit_and_nonexecuting():
    root = Path(os.environ['MURPHY_SOURCE'])
    result = subprocess.run([sys.executable, str(root / 'scripts/prepare.py'), '--help'],
                            capture_output=True, text=True, timeout=10, check=True)
    help_text = ' '.join(result.stdout.split())
    assert '--policy policies/murphy.v4.json --profile full' in help_text
    assert 'Preparation does not launch evaluation.' in help_text
    assert 'murphy.v1.json, not latest' in help_text
    assert DEFAULT == root / 'policies/murphy.v1.json'
    policy, _ = load(root / 'policies/murphy.v4.json')
    assert policy['request']['endpoints'] == ['/completion', '/v1/chat/completions']
    assert policy['request']['repeats'] == 2
