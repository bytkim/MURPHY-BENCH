import pytest
from murphy.verifiers.exact_match import verify


@pytest.mark.parametrize("answer,expected,passed", [("€", "€", True), ("👩💻", "👩\u200d💻", False),
                                                      ("A", "a", False), ("e\u0301", "é", False), (" A ", "A", False)])
def test_exact_match(answer, expected, passed):
    assert verify(answer, expected, {"normalizer": {"mode": "identity"}}) is passed
