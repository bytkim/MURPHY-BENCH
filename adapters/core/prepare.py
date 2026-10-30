"""Policy-driven, offline dataset preparation; no serving or provisioning."""
from pathlib import Path
from tokenizers import Tokenizer
from adapters.datasets import belebele, flores_plus, unicode_emoji
from adapters.core.common import Sources, aligned_ids
from adapters.core.policy import DEFAULT as DEFAULT_POLICY, allocation, load as load_policy
from adapters.util.prompt_templates import load as load_prompt, wrapper_for
from adapters.util.tokenizer_probe import measure, write_probe
from murphy.util.io import JsonlWriter, digest, file_hash, load_build, strict_loads, write_json

PROMPTS = ("flores_plus", "belebele", "unicode_emoji")


def parse_prompt_revisions(specifications):
    result = {}
    for specification in specifications:
        name, separator, revision = specification.partition("=")
        if (not separator or name not in PROMPTS or name in result
                or not revision.isascii() or not revision.isdecimal() or int(revision) < 1):
            raise ValueError("Use each known prompt name once as NAME=POSITIVE_REVISION")
        result[name] = int(revision)
    return result


def prepare(source_root, model_dir, output, *, policy_path=DEFAULT_POLICY, profile="trial", prompt_revisions=None):
    policy, policy_identity = load_policy(policy_path)
    if profile not in policy["profiles"]:
        raise ValueError("Unknown build profile")
    selected = policy["profiles"][profile]
    datasets = policy["datasets"]
    languages = policy["languages"]["fixed"]
    seed = policy["selection"]["seed"]
    emoji_pool = datasets["unicode_emoji"]["pool_per_family"]
    prompt_revisions = {} if prompt_revisions is None else prompt_revisions
    if any(name not in PROMPTS or revision != datasets[name]["prompt_revision"]
           for name, revision in prompt_revisions.items()):
        raise ValueError("Prompt revisions must match the explicit policy; select a revised policy")
    templates = {name: load_prompt(name, datasets[name]["prompt_revision"]) for name in PROMPTS}
    sources, model_dir = Sources(source_root), Path(model_dir)
    tokenizer_data = strict_loads((model_dir / "tokenizer.json").read_text())
    if tokenizer_data["normalizer"] != {"type": "NFC"} or tokenizer_data["decoder"]["type"] != "ByteLevel":
        raise ValueError("Unqualified source tokenizer normalization/decoder")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    flores = flores_plus.load(sources, revision=datasets["flores_plus"]["revision"],
                             split=datasets["flores_plus"]["split"], languages=languages)
    bele = belebele.load(sources, revision=datasets["belebele"]["revision"], languages=languages)
    emoji = unicode_emoji.load(sources, version=datasets["unicode_emoji"]["version"], pool_per_family=emoji_pool, seed=seed)
    fids = aligned_ids(flores, datasets["flores_plus"]["probe_ids"], "flores", seed=seed)
    bids = aligned_ids(bele, datasets["belebele"]["probe_ids"], "belebele", seed=seed)
    stats = {
        "flores_plus": {lang: measure([table[i]["text"] for i in fids], tokenizer) for lang, table in flores.items()},
        "belebele": {lang: measure([belebele.rendered(table[i])[0] for i in bids], tokenizer) for lang, table in bele.items()},
        "unicode_emoji": {name: measure([r["sequence"] for r in rows[:emoji_pool]], tokenizer, sequences=True) for name, rows in emoji.items()},
    }
    probe = [{"source": "flores_plus", "group": lang, "text": table[i]["text"]}
             for lang, table in flores.items() for i in fids]
    probe += [{"source": "belebele", "group": lang, "text": belebele.rendered(table[i])[0]}
              for lang, table in bele.items() for i in bids]
    probe += [{"source": "unicode_emoji", "group": name, "text": r["sequence"]}
              for name, rows in emoji.items() for r in rows[:emoji_pool]]
    for row in probe:
        row["hf_ids"] = tokenizer.encode(row["text"], add_special_tokens=False).ids
    chinese = {}
    for script in ("Hans", "Hant"):
        counts = [stats["flores_plus"]["cmn_" + script], stats["belebele"]["zho_" + script]]
        chinese[script] = sum(r["split_characters"] for r in counts) / sum(r["characters"] for r in counts)
    # Selection measures boundary stress, not model quality.
    script = "Hant" if chinese["Hant"] > chinese["Hans"] else "Hans"
    grouping = datasets["unicode_emoji"]["grouping"]
    membership = None
    if grouping["strategy"] == "single-entry-v1":
        emoji_tasks = list(unicode_emoji.tasks(emoji, selected["emoji_tasks"] // len(unicode_emoji.FAMILIES),
                                              prompt_template=templates["unicode_emoji"]))
    else:
        grouped = list(unicode_emoji.grouped_tasks(emoji, selected["emoji_tasks"], pool_per_family=emoji_pool,
                       entries_per_family=grouping["entries_per_family"], seed=seed, prompt_template=templates["unicode_emoji"]))
        emoji_tasks = [task for task, _ in grouped]
        membership = {task["task_id"]: members for task, members in grouped}
    tasks = (list(flores_plus.tasks(flores, fids[:selected["flores_ids"]], script,
                                  prompt_template=templates["flores_plus"], languages=languages))
             + list(belebele.tasks(bele, bids[:selected["belebele_ids"]], script,
                                  prompt_template=templates["belebele"], languages=languages))
             + emoji_tasks)
    counts = allocation(policy, profile)
    if len(tasks) != sum(counts.values()):
        raise ValueError("Actual task allocation differs from policy")
    wrapper = wrapper_for((model_dir / "chat_template.jinja").read_text(), [t["prompt"] for t in tasks])
    endpoints = policy["request"].get("endpoints", ["/completion"])
    names = {"/completion": "native-nonthinking", "/v1/chat/completions": "chat-nonthinking"}
    conditions = [{"id": names[endpoint], "task_ids": [t["task_id"] for t in tasks],
                   "endpoint": endpoint, "stop": [], "max_tokens": policy["request"]["max_tokens"],
                   "reasoning": "off", "literal_grammar": False, "grade": True,
                   **({"prompt_wrapper": wrapper} if endpoint == "/completion" else {})}
                  for endpoint in endpoints]
    source = {"kind": "canonical-preserved-sources", "model": policy["model"]["repository"], "model_revision": policy["model"]["revision"],
              "policy": policy_identity, "profile": profile,
              "prompt_templates": {name: template.identity for name, template in templates.items()},
              "source_files": sources.used, "source_manifest_sha256": file_hash(sources.root / "preservation/source-manifest.json"),
              "model_metadata_sha256": {p.name: file_hash(p) for p in sorted(model_dir.iterdir()) if p.is_file()},
              "selection": {"seed": seed, "order": policy["selection"]["method"], "flores_probe_ids": fids,
                            "belebele_probe_ids": bids, "selected_flores_ids": fids[:selected["flores_ids"]], "selected_belebele_ids": bids[:selected["belebele_ids"]],
                            "emoji_probe_sequences": {name: [r["sequence"] for r in rows[:emoji_pool]] for name, rows in emoji.items()}},
              "tokenizer_probe": stats, "chinese_decision": {"script": script, "pooled_character_split_rates": chinese,
                    "criterion": "higher split-character rate across aligned FLORES dev and Belebele probe; ties Hans",
                    "supersedes": "initial provisional Simplified choice; no source transliteration"},
              "probe_definition": "HF tokenizer in full source-text context; NFC byte boundaries; distinct split Unicode code points / code points; whole emoji sequences separately",
              "allocation": counts,
              "limits": ["HF tokenization measured locally; exported llama.cpp tokenization parity required before serving trial",
                         f"{profile} profile; task generation alone is not serving qualification; no forced-answer grammar"]}
    if membership is not None:
        source["selection"]["emoji_groups"] = membership
    source["tokenizer_probe_inputs_sha256"] = digest(probe)
    # Preserve historical native-only build identities; new matrices bind every condition.
    condition_identity = ({"condition": conditions[0]} if endpoints == ["/completion"]
                          else {"conditions": conditions})
    build_id = "qwen35-9b-" + profile + "-" + digest({"tasks": tasks, **condition_identity, "source": source})[:12]
    destination = Path(output) / "phase4" / build_id
    destination.mkdir(parents=True, exist_ok=False)
    with JsonlWriter(destination / "tasks.jsonl", "task") as writer:
        for t in tasks:
            writer.append(t)
    write_json(destination / "manifest.json", {"schema_version": "1", "dataset": "phase4", "build_id": build_id,
               "tasks_sha256": file_hash(destination / "tasks.jsonl"), "task_count": len(tasks), "source": source,
               "conditions": conditions, "modes": ["nonstream", "stream"], "repeats": policy["request"]["repeats"],
               "sampling": {"temperature": 0, "seed": 42, "concurrency": 1}}, "manifest")
    load_build(destination)
    write_probe(destination / "tokenizer-probe.json", probe)
    return destination
