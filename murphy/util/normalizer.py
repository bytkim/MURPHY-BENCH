"""Answer-only cleanup; never applied to token or probability evidence."""


def normalize(text, settings):
    if settings["mode"] == "identity":
        return text
    if settings["mode"] == "strip":
        return text.strip()
    raise ValueError("Unknown normalizer")
