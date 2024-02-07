import pytest

from indigo_league.utils.str_helpers import format_str


@pytest.mark.parametrize("s_1,s_2", [("Test", "T- EST")])
def test_format_str(s_1: str, s_2: str):
    assert format_str(s_1) == format_str(s_2)
