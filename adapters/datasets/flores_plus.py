"""Preserved FLORES+ dev sentences, aligned by original integer ID."""
from adapters.core.common import FLORES_REV, LANGUAGES, index_rows, task
from adapters.util.prompt_templates import load as load_prompt


def load(sources, *, revision=FLORES_REV, split="dev", languages=LANGUAGES):
    return {lang: index_rows(sources.rows(f"sources/flores_plus/{revision}/{split}/{lang}.jsonl"),
                             lambda row: row["id"])
            for lang in (*languages, "cmn_Hans", "cmn_Hant")}


def tasks(tables, ids, script, *, prompt_template=None, languages=LANGUAGES):
    prompt_template = prompt_template or load_prompt("flores_plus")
    for identity in ids:
        for lang in (*languages, "cmn_" + script):
            text = tables[lang][identity]["text"]
            yield task(f"flores-{identity}-{lang}", prompt_template.render(text=text), text)
