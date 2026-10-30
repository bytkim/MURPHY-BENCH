import pytest
from murphy.util.normalizer import normalize


def test_identity_is_exact():
    value = " \t€👩\u200d💻\n"
    assert normalize(value, {"mode": "identity"}) == value


def test_strip_does_not_change_unicode_or_case():
    assert normalize(" A e\u0301 ", {"mode": "strip"}) == "A e\u0301"


def test_unknown_fails_closed():
    with pytest.raises(ValueError):
        normalize("text", {"mode": "other"})
