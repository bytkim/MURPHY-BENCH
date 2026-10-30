"""Original Belebele, aligned by passage link plus question number."""
from adapters.core.common import BELEBELE_REV, LANGUAGES, index_rows, task
from adapters.util.prompt_templates import load as load_prompt
from murphy.util.io import digest


def load(sources, *, revision=BELEBELE_REV, languages=LANGUAGES):
    return {lang: index_rows(sources.rows(f"sources/belebele/{revision}/data/{lang}.jsonl"),
                             lambda row: (row["link"], row["question_number"]))
            for lang in (*languages, "zho_Hans", "zho_Hant")}


def rendered(row):
    choices = [row[f"mc_answer{i}"] for i in range(1, 5)]
    index = int(row["correct_answer_num"]) - 1
    if index not in range(4):
        raise ValueError("Invalid Belebele answer key")
    text = row["flores_passage"] + "\n\n" + row["question"] + "\n" + "\n".join(
        f"{letter}. {choice}" for letter, choice in zip("ABCD", choices))
    expected = f"{'ABCD'[index]}. {choices[index]}"
    return text, expected


def tasks(tables, ids, script, *, prompt_template=None, languages=LANGUAGES):
    prompt_template = prompt_template or load_prompt("belebele")
    for identity in ids:
        for lang in (*languages, "zho_" + script):
            text, expected = rendered(tables[lang][identity])
            yield task(f"belebele-{digest(identity)[:16]}-{lang}", prompt_template.render(text=text), expected)
