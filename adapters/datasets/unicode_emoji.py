"""Disjoint Unicode families with explicit single-entry or balanced grouping."""
import re

from adapters.core.common import task
from adapters.util.prompt_templates import load as load_prompt
from murphy.util.io import digest

FAMILIES = ("flags-including-tags", "zwj", "modifier-without-zwj", "single-code-point", "other-multi-code-point")


def family(sequence):
    points = list(map(ord, sequence))
    if (len(points) == 2 and all(0x1F1E6 <= p <= 0x1F1FF for p in points)) or any(0xE0020 <= p <= 0xE007F for p in points):
        return FAMILIES[0]
    if 0x200D in points:
        return FAMILIES[1]
    if any(0x1F3FB <= p <= 0x1F3FF for p in points):
        return FAMILIES[2]
    if len(points) == 1:
        return FAMILIES[3]
    if 0x20E3 not in points and not any(p in (0xFE0E, 0xFE0F) for p in points):
        raise ValueError("Unclassified multi-code-point emoji")
    return FAMILIES[4]


def load(sources, *, version="17.0.0", pool_per_family=60, seed=42):
    text = sources.text(f"sources/unicode_emoji/{version}/emoji/emoji-test.txt")
    groups = {name: [] for name in FAMILIES}
    seen = set()
    for line in text.splitlines():
        match = re.match(r"^([0-9A-F ]+)\s*; fully-qualified\s*# \S+ E[0-9.]+ (.+)$", line)
        if match:
            sequence = "".join(chr(int(p, 16)) for p in match[1].split())
            if sequence in seen:
                raise ValueError("Duplicate emoji sequence")
            seen.add(sequence)
            groups[family(sequence)].append({"sequence": sequence, "name": match[2]})
    if any(len(rows) < pool_per_family for rows in groups.values()):
        raise ValueError("Each emoji family needs the requested unique sequence pool")
    return {name: sorted(rows, key=lambda row: digest([seed, "emoji", row["sequence"]])) for name, rows in groups.items()}


def tasks(groups, count, *, prompt_template=None):
    prompt_template = prompt_template or load_prompt("unicode_emoji")
    for name, rows in groups.items():
        for row in rows[:count]:
            sequence, english = row["sequence"], row["name"]
            yield task(f"emoji-{name}-{digest(sequence)[:12]}",
                       prompt_template.render(sequence=sequence, name=english),
                       "{" + sequence + " | " + english + "}")


def grouped_tasks(groups, count, *, pool_per_family, entries_per_family, seed, prompt_template):
    """Each complete cycle uses every selected sequence once per family."""
    if (type(count) is not int or count < 1 or type(entries_per_family) is not int
            or entries_per_family < 1 or type(pool_per_family) is not int
            or pool_per_family < 1 or pool_per_family % entries_per_family):
        raise ValueError("Invalid balanced emoji grouping")
    if set(groups) != set(FAMILIES):
        raise ValueError("Missing or unknown emoji family")
    pool = {name: groups[name][:pool_per_family] for name in FAMILIES}
    sequences = [r["sequence"] for rows in pool.values() for r in rows]
    if (len(sequences) != pool_per_family * len(FAMILIES) or len(set(sequences)) != len(sequences)
            or any(family(r["sequence"]) != name for name, rows in pool.items() for r in rows)):
        raise ValueError("Incomplete, duplicate or misclassified emoji pool")
    per_cycle = pool_per_family // entries_per_family
    seen = set()
    for index in range(count):
        cycle, slot = divmod(index, per_cycle)
        if slot == 0:
            shuffled = {name: sorted(rows, key=lambda row: digest([seed, "emoji-group-cycle", cycle, name, row["sequence"]]))
                        for name, rows in pool.items()}
        members = [{"source_id": f"emoji-{name}-{digest(row['sequence'])[:12]}", "family": name, **row}
                   for name in FAMILIES
                   for row in shuffled[name][slot * entries_per_family:(slot + 1) * entries_per_family]]
        members.sort(key=lambda row: digest([seed, "emoji-group-order", index, row["sequence"]]))
        identity = "emoji-group-" + digest([r["source_id"] for r in members])[:16]
        if identity in seen:
            raise ValueError("Duplicate emoji group; do not silently resample")
        seen.add(identity)
        expected = "\n".join("{" + r["sequence"] + " | " + r["name"] + "}" for r in members)
        yield task(identity, prompt_template.render(count=str(len(members)), entries=expected), expected), members
