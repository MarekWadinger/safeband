"""Tests for malformed-input handling in RpcOutlierDetector.preprocess."""

import logging
import sys
from pathlib import Path
from typing import cast

import pytest
from paho.mqtt.client import MQTTMessage

sys.path.insert(1, str(Path(__file__).parent.parent))

from rpc_server import RpcOutlierDetector


class TestPreprocessUnparsable:
    """Unparsable payloads are skipped with a warning instead of raising."""

    def test_preprocess_nonnumeric_bytes_returns_none(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Bytes that do not parse as float yield None and a warning."""
        with caplog.at_level(logging.WARNING, logger="rpc_server"):
            result = RpcOutlierDetector().preprocess(b"junk", ["plant/a"])

        assert result is None
        assert "unparsable" in caplog.text

    def test_preprocess_nonnumeric_mqtt_payload_returns_none(self) -> None:
        """An MQTT message with a non-numeric payload yields None."""
        msg = MQTTMessage(topic=b"plant/a")
        msg.payload = b"not-a-number"

        assert RpcOutlierDetector().preprocess(msg, ["plant/a"]) is None

    def test_preprocess_nonnumeric_dict_value_returns_none(self) -> None:
        """A dict whose topic value does not parse as float yields None."""
        result = RpcOutlierDetector().preprocess(
            {"plant/a": "not-a-number"},
            ["plant/a"],
        )

        assert result is None

    def test_preprocess_unknown_type_returns_none(self) -> None:
        """An unsupported input type yields None instead of raising."""
        assert (
            RpcOutlierDetector().preprocess(cast("bytes", 42), ["plant/a"])
            is None
        )

    def test_preprocess_numeric_bytes_returns_record(self) -> None:
        """Valid numeric bytes still produce a time/data record."""
        result = RpcOutlierDetector().preprocess(b"21.0", ["plant/a"])

        assert result is not None
        assert result["data"] == {"plant/a": 21.0}
