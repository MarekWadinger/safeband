"""Tests for argument parsing and configuration building."""

from argparse import Namespace
from configparser import ConfigParser
from typing import NotRequired

import pytest

from functions.parse import _to_bool, build_config, get_valid_type


class TestGetValidType:
    """Tests for unwrapping type hints into concrete Python types."""

    def test_not_required_bool_unwraps_to_bool(self) -> None:
        """NotRequired[bool] resolves to bool, not str."""
        assert get_valid_type(NotRequired[bool]) is bool

    def test_not_required_str_unwraps_to_str(self) -> None:
        """NotRequired[str] resolves to str."""
        assert get_valid_type(NotRequired[str]) is str

    def test_not_required_generic_unwraps_to_generic(self) -> None:
        """NotRequired[dict[str, int]] resolves to dict[str, int]."""
        assert get_valid_type(NotRequired[dict[str, int]]) == dict[str, int]


class TestToBool:
    """Tests for explicit boolean parsing of config values."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, True),
            (False, False),
            ("True", True),
            ("true", True),
            ("FALSE", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
        ],
    )
    def test_valid_values(self, value: object, expected: bool) -> None:
        """Bools pass through; common string spellings are parsed."""
        assert _to_bool(value) is expected

    def test_invalid_value_raises(self) -> None:
        """Unrecognised values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid boolean value"):
            _to_bool("maybe")


class TestBuildConfigDebug:
    """Tests for debug flag precedence between CLI args and config file."""

    @staticmethod
    def make_parser(debug: str) -> ConfigParser:
        """Build a hermetic ConfigParser with a [setup] debug entry."""
        config_parser = ConfigParser()
        config_parser["setup"] = {"debug": debug}
        return config_parser

    def test_config_false_yields_real_bool_false(self) -> None:
        """[setup] debug=False with no CLI flag yields the bool False."""
        args = Namespace(debug=None)
        config = build_config(args, self.make_parser("False"))
        assert config["setup"]["debug"] is False

    def test_config_true_yields_real_bool_true(self) -> None:
        """[setup] debug=True with no CLI flag yields the bool True."""
        args = Namespace(debug=None)
        config = build_config(args, self.make_parser("True"))
        assert config["setup"]["debug"] is True

    def test_args_true_overrides_config_false(self) -> None:
        """CLI --debug overrides a config-file debug=False."""
        args = Namespace(debug=True)
        config = build_config(args, self.make_parser("False"))
        assert config["setup"]["debug"] is True
