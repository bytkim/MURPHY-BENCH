"""Ten handwritten tasks; request conditions stay in the build manifest."""
from pathlib import Path

from murphy.util.io import JsonlWriter, digest, file_hash, load_build, write_json

TEXTS = {
    "ascii": "AB", "two-byte": "A\u07ffB", "euro": "A€B",
    "repeat-euro": "A€€B", "emoji": "A🙂B", "zwj": "👩\u200d💻",
    "keycap": "1️⃣", "flag": "🇯🇵", "mid-character": "€", "chinese": "中文",
}


def build(root):
    tasks = [
        {"schema_version": "1", "task_id": key, "prompt": f"Repeat exactly: {text}",
         "expected": text, "verifier": {"name": "exact_match", "settings": {"normalizer": {"mode": "identity"}}}}
        for key, text in TEXTS.items()
    ]
    def condition(name, ids, **options):
        return {"id": name, "task_ids": ids, "endpoint": "/completion", "stop": [],
                "max_tokens": 64, "reasoning": "off", "literal_grammar": True, "grade": True, **options}
    conditions = [
        condition("literal", list(TEXTS)),
        condition("stop-euro", ["euro"], stop=["€"], grade=False),
        condition("length-byte-1", ["mid-character"], max_tokens=1, grade=False),
        condition("length-byte-2", ["mid-character"], max_tokens=2, grade=False),
        condition("chat", ["euro", "emoji"], endpoint="/v1/chat/completions"),
        condition("chat-reasoning", ["euro"], endpoint="/v1/chat/completions", reasoning="on"),
    ]
    build_id = "fixture-v1-" + digest({"tasks": tasks, "conditions": conditions})[:12]
    destination = Path(root) / "fixture" / build_id
    destination.mkdir(parents=True, exist_ok=False)
    with JsonlWriter(destination / "tasks.jsonl", "task") as writer:
        for task in tasks:
            writer.append(task)
    manifest = {"schema_version": "1", "dataset": "fixture", "build_id": build_id,
                "tasks_sha256": file_hash(destination / "tasks.jsonl"), "task_count": len(tasks),
                "source": {"kind": "handwritten", "version": "1", "not_benchmark_tasks": True},
                "conditions": conditions, "modes": ["nonstream", "stream"], "repeats": 2,
                "sampling": {"temperature": 0, "seed": 42, "concurrency": 1}}
    write_json(destination / "manifest.json", manifest, "manifest")
    load_build(destination)
    return destination
