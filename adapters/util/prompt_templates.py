"""Small, explicit prompt revisions; values are substituted once, never evaluated."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from string import Template
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[1] / "datasets/prompts"

def wrapper_for(template, prompts):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def fail(message):
        raise ValueError(message)
    env.globals["raise_exception"] = fail
    compiled = env.from_string(template)
    def render(prompt):
        return compiled.render(messages=[{"role": "user", "content": prompt}],
                               add_generation_prompt=True, enable_thinking=False)
    marker = "MURPHY_PROMPT_BOUNDARY_0b8612"
    rendered = render(marker)
    if rendered.count(marker) != 1:
        raise ValueError("Template is not a simple single-turn wrapper")
    prefix, suffix = rendered.split(marker)
    if any(render(p) != prefix + p + suffix for p in prompts):
        raise ValueError("Template wrapper differs from actual prompt rendering")
    return {"prefix": prefix, "suffix": suffix}


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    revision: int
    raw: bytes

    @property
    def identity(self):
        return {"revision": self.revision, "sha256": hashlib.sha256(self.raw).hexdigest()}

    def render(self, **values):
        # The one terminal file LF is a storage convention, not part of the prompt.
        template = Template(self.raw.decode("utf-8").removesuffix("\n"))
        if not template.is_valid() or set(template.get_identifiers()) != set(values):
            raise ValueError("Prompt template fields do not match supplied values")
        if any(not isinstance(v, str) for v in values.values()):
            raise ValueError("Prompt template values must be strings")
        return template.substitute(values)


def load(name, revision=1):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or type(revision) is not int or revision < 1:
        raise ValueError("Invalid prompt name/revision")
    # Never silently fall back to another revision.
    return PromptTemplate(name, revision, (ROOT / name / f"{name}-v{revision}.txt").read_bytes())
