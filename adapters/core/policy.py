"""One validated recipe snapshot; no implicit latest policy or configuration merging."""
from pathlib import Path
import hashlib

from adapters.datasets.unicode_emoji import FAMILIES
from murphy.util.io import digest, strict_loads, validate

DEFAULT = Path(__file__).resolve().parents[2] / "policies/murphy.v1.json"


def check(policy):
    validate(policy, "policy")
    datasets = policy["datasets"]
    pool = datasets["unicode_emoji"]["pool_per_family"]
    grouping = datasets["unicode_emoji"]["grouping"]
    if grouping["strategy"] == "balanced-families-v1":
        size = grouping["entries_per_family"]
        if pool % size:
            raise ValueError("Emoji pool must partition into whole family groups")
        if policy["profiles"]["full"]["emoji_tasks"] % (pool // size):
            raise ValueError("Full emoji profile must contain whole balanced cycles")
    for profile in policy["profiles"].values():
        if (profile["flores_ids"] > datasets["flores_plus"]["probe_ids"]
                or profile["belebele_ids"] > datasets["belebele"]["probe_ids"]):
            raise ValueError("Task selection exceeds aligned probe pool")
        if grouping["strategy"] == "single-entry-v1" and (profile["emoji_tasks"] % len(FAMILIES) or profile["emoji_tasks"] > pool * len(FAMILIES)):
            raise ValueError("Single-entry emoji tasks must balance within the selected pool")
    if any(policy["profiles"]["trial"][k] > policy["profiles"]["full"][k] for k in ("flores_ids", "belebele_ids", "emoji_tasks")):
        raise ValueError("Trial selection exceeds full selection")
    return policy


def load(path=DEFAULT):
    raw = Path(path).read_bytes()
    policy = check(strict_loads(raw))
    return policy, {"policy_id": policy["policy_id"], "revision": policy["revision"],
                    "sha256": hashlib.sha256(raw).hexdigest(), "snapshot_sha256": digest(policy), "snapshot": policy}


def allocation(policy, profile):
    selected = policy["profiles"][profile]
    languages = len(policy["languages"]["fixed"]) + 1
    return {"flores_plus": selected["flores_ids"] * languages,
            "belebele": selected["belebele_ids"] * languages, "unicode_emoji": selected["emoji_tasks"]}
