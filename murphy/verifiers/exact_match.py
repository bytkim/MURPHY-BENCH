"""Single-turn exact answer comparison."""
from murphy.util.normalizer import normalize


def verify(answer, expected, settings):
    return normalize(answer, settings["normalizer"]) == normalize(expected, settings["normalizer"])
