"""Tests for transport wiring and stop detection in RpcOutlierDetector."""

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


class TestRawSourceStopDetection:
    """run() polls the raw source node captured by get_source."""

    def test_get_source_mqtt_wrapped_keeps_raw_reference(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The raw MQTT node is kept even though the source is wrapped."""
        raw = Stream()
        from_mqtt = MagicMock(return_value=raw)
        monkeypatch.setattr(Stream, "from_mqtt", from_mqtt)
        detector = RpcOutlierDetector()

        source = detector.get_source(
            {"host": "broker", "port": 1883},
            ["topic_a"],
            debug=False,
        )

        assert detector._raw_source is raw
        # The returned pipeline node is the accumulate/filter wrapper.
        assert source is not raw

    def test_run_raw_source_stopped_returns(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run() exits via the raw source flag, not upstream probing."""
        monkeypatch.setattr(
            rpc_server.time,
            "sleep",
            MagicMock(side_effect=AssertionError("should not sleep")),
        )
        detector = RpcOutlierDetector()
        detector._raw_source = MagicMock(stopped=True)
        pipeline = MagicMock()

        # A bare Stream has no ``stopped`` and no usable upstream chain;
        # passing it proves run() does not probe the wrapped source.
        detector.run(
            {"host": "broker", "port": 1883},
            Stream(),
            pipeline,
            debug=False,
        )

        pipeline.start.assert_called_once()
