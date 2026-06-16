"""Tests for transport wiring and stop detection in RpcOutlierDetector."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
from streamz import Stream

sys.path.insert(1, str(Path(__file__).parent.parent))

import rpc_server
from rpc_server import RpcOutlierDetector

if TYPE_CHECKING:
    from functions.typing_extras import KafkaClient, NATSClient


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


class TestGetSourceKafka:
    """The Kafka group.id default must not clobber user configuration."""

    def test_kafka_source_user_group_id_wins(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A user-supplied group.id is passed through unchanged."""
        sentinel = Stream()
        from_kafka = MagicMock(return_value=sentinel)
        monkeypatch.setattr(Stream, "from_kafka", from_kafka)

        # The "group.id" key is extra w.r.t. the KafkaClient TypedDict,
        # mirroring how confluent-kafka options arrive from config files.
        config = cast(
            "KafkaClient",
            {"bootstrap_servers": "localhost:9092", "group.id": "my_group"},
        )
        source = RpcOutlierDetector().get_source(
            config,
            ["topic_a"],
            debug=False,
        )

        from_kafka.assert_called_once_with(
            ["topic_a"],
            {
                "bootstrap_servers": "localhost:9092",
                "group.id": "my_group",
            },
        )
        assert source is sentinel

    def test_kafka_source_no_group_id_gets_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without a configured group.id, the service default applies."""
        from_kafka = MagicMock(return_value=Stream())
        monkeypatch.setattr(Stream, "from_kafka", from_kafka)

        RpcOutlierDetector().get_source(
            {"bootstrap_servers": "localhost:9092"},
            ["topic_a"],
            debug=False,
        )

        config = from_kafka.call_args[0][1]
        assert config["group.id"] == "detection_service"


class TestGetSourceNats:
    """A NATSClient config wires from_nats with the MQTT-style wrapping."""

    def test_nats_source_wires_from_nats_and_wraps(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """from_nats is built, then wrapped with accumulate/filter."""
        raw = Stream()
        from_nats = MagicMock(return_value=raw)
        monkeypatch.setattr(Stream, "from_nats", from_nats)
        detector = RpcOutlierDetector()

        config: NATSClient = {"servers": "nats://localhost:4222"}
        source = detector.get_source(config, ["topic_a"], debug=False)

        from_nats.assert_called_once_with(
            servers="nats://localhost:4222",
            topic=["topic_a"],
        )
        # The raw broker node is captured for stop detection, and the
        # returned node is the accumulate/filter wrapper, just like MQTT.
        assert detector._raw_source is raw
        assert source is not raw


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
