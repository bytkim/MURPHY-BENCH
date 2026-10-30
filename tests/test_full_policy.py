import copy
from adapters.util.prompt_templates import wrapper_for
from pathlib import Path
import os

import pytest

from adapters.datasets.belebele import rendered, tasks
from adapters.core.policy import load
from adapters.util.prompt_templates import load as prompt


def test_full_policy_changes_only_belebele_wording():
    root = Path(os.environ['MURPHY_SOURCE'])
    old, _ = load(root / 'policies/murphy.v2.json')
    new, _ = load(root / 'policies/murphy.v3.json')
    expected = copy.deepcopy(old)
    expected['revision'] = 3
    expected['datasets']['belebele']['prompt_revision'] = 2
    assert new == expected


@pytest.mark.parametrize('whitespace', ['  ', '\t\n', '\u00a0', '\u3000'])
@pytest.mark.parametrize('answer', range(1, 5))
def test_source_whitespace_and_answers_survive_template_trim(whitespace, answer):
    row = {'flores_passage': whitespace + 'passage' + whitespace, 'question': 'question',
           'correct_answer_num': str(answer),
           **{f'mc_answer{i}': f'choice {i}' + whitespace for i in range(1, 5)}}
    text, expected = rendered(row)
    output = prompt('belebele', 2).render(text=text)
    assert text in output and expected == f'{"ABCD"[answer - 1]}. choice {answer}' + whitespace
    wrapper_for('U{{ messages[0].content | trim }}A', [output])
    tables = {lang: {'id': row} for lang in ('eng_Latn', 'zho_Hant')}
    before = list(tasks(tables, ['id'], 'Hant', languages=['eng_Latn']))
    after = list(tasks(tables, ['id'], 'Hant', languages=['eng_Latn'], prompt_template=prompt('belebele', 2)))
    assert [{k: v for k, v in t.items() if k != 'prompt'} for t in before] == [
        {k: v for k, v in t.items() if k != 'prompt'} for t in after]


def test_old_wording_still_rejects_trimmed_source():
    with pytest.raises(ValueError, match='differs'):
        wrapper_for('{{ messages[0].content | trim }}',
                    [prompt('belebele', 1).render(text='passage\nD. choice  ')])
