"""Tests for the Pulsar transport branch of RpcOutlierDetector.get_source."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from streamz import Stream

sys.path.insert(1, str(Path(__file__).parent.parent))

import rpc_server
from rpc_server import RpcOutlierDetector


class TestGetSourcePulsar:
    """Tests for wiring a Pulsar source without a live broker."""

    def test_pulsar_source_wires_from_pulsar(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A PulsarClient config dispatches to Stream.from_pulsar."""
        sentinel = Stream()
        from_pulsar = MagicMock(return_value=sentinel)
        monkeypatch.setattr(Stream, "from_pulsar", from_pulsar)

        source = RpcOutlierDetector().get_source(
            {"service_url": "pulsar://localhost:6650"},
            ["topic_a"],
            debug=False,
        )

        from_pulsar.assert_called_once_with(
            "pulsar://localhost:6650",
            ["topic_a"],
            subscription_name="detection_service",
        )
        assert source is sentinel

    def test_pulsar_source_raises_when_not_installed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without pulsar-client installed, get_source raises RuntimeError."""
        monkeypatch.setattr(rpc_server, "_PULSAR_AVAILABLE", False)

        with pytest.raises(
            RuntimeError,
            match="pulsar-client is not installed",
        ):
            RpcOutlierDetector().get_source(
                {"service_url": "pulsar://localhost:6650"},
                ["topic_a"],
                debug=False,
            )
