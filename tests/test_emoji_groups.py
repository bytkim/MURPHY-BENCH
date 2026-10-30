from collections import Counter
import os
from pathlib import Path

import pytest

from adapters.core.policy import check, load
from adapters.util.prompt_templates import load as template
from adapters.datasets.unicode_emoji import FAMILIES, grouped_tasks


def groups():
    sequences = [["🇯🇵", "🇺🇸", "🇬🇧", "🇨🇦"], ["👩‍💻", "👨‍💻", "👩‍🔬", "👨‍🔬"],
                 ["👍🏻", "👍🏼", "👍🏽", "👍🏾"], ["😀", "😃", "😄", "😁"], ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]]
    return {name: [{"sequence": s, "name": f"name {i}"} for i, s in enumerate(rows)] for name, rows in zip(FAMILIES, sequences)}


def generate(source=None, count=8, **kwargs):
    return list(grouped_tasks(groups() if source is None else source, count, pool_per_family=4,
                             entries_per_family=2, seed=kwargs.get("seed", 42), prompt_template=template("unicode_emoji", 3)))


def test_balanced_cycles_exact_answers_and_membership():
    result = generate()
    assert len({t["task_id"] for t, _ in result}) == len(result)
    exposure = Counter()
    for task, members in result:
        assert Counter(m["family"] for m in members) == {family: 2 for family in FAMILIES}
        assert len({m["sequence"] for m in members}) == 10
        exposure.update(m["source_id"] for m in members)
        expected = "\n".join("{" + m["sequence"] + " | " + m["name"] + "}" for m in members)
        assert task["expected"] == expected and task["prompt"].endswith(expected)
        assert "following 10 lines exactly" in task["prompt"]
        assert task["verifier"]["settings"]["normalizer"]["mode"] == "identity"
    assert set(exposure.values()) == {4}  # 8 groups / 2 groups per cycle.
    assert len(exposure) == 20


def test_stable_prefix_mapping_order_and_seed():
    assert generate(count=3) == generate()[:3]
    assert generate(dict(reversed(list(groups().items())))) == generate()
    assert generate(seed=43) != generate()


@pytest.mark.parametrize("failure", ["missing", "duplicate", "misclassified", "short"])
def test_bad_pools_rejected(failure):
    source = groups()
    if failure == "missing":
        source.pop(FAMILIES[0])
    elif failure == "duplicate":
        source[FAMILIES[0]][1] = source[FAMILIES[0]][0]
    elif failure == "misclassified":
        source[FAMILIES[0]][0]["sequence"] = "😎"
    else:
        source[FAMILIES[0]].pop()
    with pytest.raises(ValueError):
        generate(source)


def test_group_policy_balance_rules():
    path = Path(os.environ['MURPHY_SOURCE']) / "policies/murphy.v2.json"
    policy, _ = load(path)
    policy["profiles"]["full"]["emoji_tasks"] = 301
    with pytest.raises(ValueError, match="whole balanced cycles"):
        check(policy)
    policy, _ = load(path)
    policy["datasets"]["unicode_emoji"]["grouping"]["entries_per_family"] = 7
    with pytest.raises(ValueError, match="partition"):
        check(policy)
