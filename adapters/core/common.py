"""Shared source admission and deterministic selection, outside the core."""
from pathlib import Path

from murphy.util.io import digest, file_hash, strict_loads

FLORES_REV = "1147e2dc59f37fbb5c22fd8478770de02cd399f9"
BELEBELE_REV = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
LANGUAGES = ("eng_Latn", "rus_Cyrl", "spa_Latn", "arb_Arab", "jpn_Jpan")


class Sources:
    def __init__(self, root):
        self.root = Path(root)
        self.inventory = strict_loads((self.root / "preservation/source-manifest.json").read_text())
        self.files = {item["path"]: item for item in self.inventory["files"]}
        self.used = {}

    def text(self, relative):
        path = self.root / relative
        record = self.files[relative]
        if path.is_symlink() or file_hash(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Preserved source identity mismatch: {relative}")
        self.used[relative] = record["sha256"]
        return path.read_text(encoding="utf-8")

    def rows(self, relative):
        return [strict_loads(line) for line in self.text(relative).splitlines()]


def index_rows(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result:
            raise ValueError("Duplicate source alignment ID")
        result[identity] = row
    return result


def aligned_ids(tables, count, namespace, *, seed=42):
    ids = set(next(iter(tables.values())))
    if any(set(table) != ids for table in tables.values()):
        raise ValueError("Source alignment sets differ")
    if len(ids) < count:
        raise ValueError("Insufficient aligned source records")
    return sorted(ids, key=lambda key: digest([seed, namespace, key]))[:count]


def task(task_id, prompt, expected):
    return {"schema_version": "1", "task_id": task_id, "prompt": prompt, "expected": expected,
            "verifier": {"name": "exact_match", "settings": {"normalizer": {"mode": "identity"}}}}
