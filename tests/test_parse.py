"""Tests for argument parsing and configuration building."""

from argparse import Namespace
from configparser import ConfigParser

import pytest
from pandas import Timedelta

from functions.parse import build_config, get_valid_client


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
        assert config.setup.debug is False

    def test_config_true_yields_real_bool_true(self) -> None:
        """[setup] debug=True with no CLI flag yields the bool True."""
        args = Namespace(debug=None)
        config = build_config(args, self.make_parser("True"))
        assert config.setup.debug is True

    def test_args_true_overrides_config_false(self) -> None:
        """CLI --debug overrides a config-file debug=False."""
        args = Namespace(debug=True)
        config = build_config(args, self.make_parser("False"))
        assert config.setup.debug is True


class TestBuildConfigCoercion:
    """build_config coerces strings and normalises empty values to None."""

    def test_string_threshold_coerced_to_float(self) -> None:
        """A config-file threshold string is coerced to float."""
        parser = ConfigParser()
        parser["model"] = {"threshold": "0.5"}
        config = build_config(Namespace(), parser)
        assert config.model.threshold == 0.5

    def test_empty_string_normalised_to_none(self) -> None:
        """A "" config value is treated as absent (None)."""
        parser = ConfigParser()
        parser["setup"] = {"recovery_path": ""}
        config = build_config(Namespace(), parser)
        assert config.setup.recovery_path is None

    def test_string_timedelta_coerced(self) -> None:
        """A config-file t_e string is coerced to a pandas Timedelta."""
        parser = ConfigParser()
        parser["model"] = {"t_e": "2h"}
        config = build_config(Namespace(), parser)
        assert config.model.t_e == Timedelta("2h")


class TestGetValidClient:
    """get_valid_client enforces the exactly-one-client rule."""

    def test_single_client_resolved(self) -> None:
        """A single fully specified client is moved into ``client``."""
        config = build_config(
            Namespace(host="broker", port=1883),
            ConfigParser(),
        )
        config = get_valid_client(config)
        assert config.client is not None
        assert config.client.host == "broker"

    def test_multiple_clients_raise(self) -> None:
        """Two fully specified clients raise with the exact message."""
        config = build_config(
            Namespace(
                host="broker",
                port=1883,
                bootstrap_servers="kafka:9092",
            ),
            ConfigParser(),
        )
        with pytest.raises(
            ValueError,
            match=r"Multiple clients specified: \['mqtt', 'kafka'\]",
        ):
            get_valid_client(config)

    def test_no_client_raises(self) -> None:
        """A partial client (missing port) is absent; no client raises."""
        config = build_config(Namespace(host="broker"), ConfigParser())
        with pytest.raises(
            ValueError,
            match="Specify one of the clients",
        ):
            get_valid_client(config)
