"""Tests for the transport branches of RpcOutlierDetector.get_sink."""

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from streamz import Stream

sys.path.insert(1, str(Path(__file__).parent.parent))

import rpc_server
from rpc_server import RpcOutlierDetector

if TYPE_CHECKING:
    from functions.typing_extras import (
        FileClient,
        KafkaClient,
        MQTTClient,
        PulsarClient,
    )


class TestDumpToFile:
    """Tests for the JSON-lines file writer."""

    def test_dump_to_file_result_visible_before_close(
        self,
        tmp_path: Path,
    ) -> None:
        """Each result reaches disk immediately, without closing the file."""
        output = tmp_path / "out.json"

        with output.open("a") as f:
            RpcOutlierDetector().dump_to_file({"anomaly": 0}, f)

            assert json.loads(output.read_text().strip()) == {"anomaly": 0}


class TestGetSinkFile:
    """Tests for the file sink branch."""

    def test_file_sink_appends_json_lines(self, tmp_path: Path) -> None:
        """Emitted results are appended to the output file as JSON."""
        output = tmp_path / "out.json"
        config: FileClient = {"path": "unused.csv", "output": str(output)}
        detector = Stream()

        RpcOutlierDetector().get_sink(config, ["plant/a"], detector)
        detector.emit({"anomaly": 0})
        rpc_server._exit_stack.close()

        assert json.loads(output.read_text().strip()) == {"anomaly": 0}


class TestGetSinkMqtt:
    """Tests for the MQTT sink branch."""

    def test_mqtt_sink_uses_common_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The MQTT topic prefix is derived from the input topics."""
        to_mqtt = MagicMock()
        monkeypatch.setattr(Stream, "to_mqtt", to_mqtt)
        config: MQTTClient = {"host": "broker", "port": 1883}

        RpcOutlierDetector().get_sink(
            config,
            ["plant/a", "plant/b"],
            Stream(),
        )

        to_mqtt.assert_called_once_with(
            host="broker",
            port=1883,
            topic="plant/",
            publish_kwargs={"retain": True},
        )

    def test_mqtt_sink_honors_out_topics(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configured out_topics override the input-derived prefix."""
        to_mqtt = MagicMock()
        monkeypatch.setattr(Stream, "to_mqtt", to_mqtt)
        config: MQTTClient = {"host": "broker", "port": 1883}

        RpcOutlierDetector().get_sink(
            config,
            ["plant/a", "plant/b"],
            Stream(),
            out_topics=["custom/limits"],
        )

        to_mqtt.assert_called_once_with(
            host="broker",
            port=1883,
            topic="custom/",
            publish_kwargs={"retain": True},
        )


class TestGetSinkKafka:
    """Tests for the Kafka sink branch."""

    def test_kafka_topic_derived_from_in_topics(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without out_topics, the sink topic comes from the input prefix."""
        to_kafka = MagicMock()
        monkeypatch.setattr(Stream, "to_kafka", to_kafka)
        config: KafkaClient = {"bootstrap_servers": "localhost:9092"}

        RpcOutlierDetector().get_sink(
            config,
            ["plant/a", "plant/b"],
            Stream(),
        )

        to_kafka.assert_called_once_with("plant/dynamic_limits", config)

    def test_kafka_topic_honors_out_topics(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The first configured out_topic names the Kafka sink topic."""
        to_kafka = MagicMock()
        monkeypatch.setattr(Stream, "to_kafka", to_kafka)
        config: KafkaClient = {"bootstrap_servers": "localhost:9092"}

        RpcOutlierDetector().get_sink(
            config,
            ["plant/a", "plant/b"],
            Stream(),
            out_topics=["custom/limits"],
        )

        to_kafka.assert_called_once_with("custom/limits", config)

    def test_kafka_topic_honors_out_topics_string(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A single out_topic configured as a plain string is honored."""
        to_kafka = MagicMock()
        monkeypatch.setattr(Stream, "to_kafka", to_kafka)
        config: KafkaClient = {"bootstrap_servers": "localhost:9092"}

        RpcOutlierDetector().get_sink(
            config,
            ["plant/a"],
            Stream(),
            out_topics="custom/limits",
        )

        to_kafka.assert_called_once_with("custom/limits", config)


class TestGetSinkPulsar:
    """Tests for the Pulsar sink branch."""

    def test_pulsar_sink_wires_to_pulsar(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A PulsarClient config dispatches to Stream.to_pulsar."""
        to_pulsar = MagicMock()
        monkeypatch.setattr(Stream, "to_pulsar", to_pulsar)
        config: PulsarClient = {"service_url": "pulsar://localhost:6650"}

        RpcOutlierDetector().get_sink(config, ["plant/a"], Stream())

        assert to_pulsar.call_count == 1
        args, kwargs = to_pulsar.call_args
        assert args[0] == "pulsar://localhost:6650"
        assert args[1] == "plant/dynamic_limits"
        assert "schema" in kwargs["producer_config"]

    def test_pulsar_sink_raises_when_not_installed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without pulsar-client installed, get_sink raises RuntimeError."""
        monkeypatch.setattr(rpc_server, "_PULSAR_AVAILABLE", False)

        with pytest.raises(
            RuntimeError,
            match="pulsar-client is not installed",
        ):
            RpcOutlierDetector().get_sink(
                {"service_url": "pulsar://localhost:6650"},
                ["plant/a"],
                Stream(),
            )
