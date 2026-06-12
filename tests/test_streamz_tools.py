"""Tests for the custom map stream operator and the MQTT sink."""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import paho.mqtt.client as mqtt
import pytest
from streamz import Stream

sys.path.insert(1, str(Path(__file__).parent.parent))

# Importing registers the custom ``map`` operator on Stream.
import functions.streamz_tools  # noqa: F401
from functions.streamz_tools import to_mqtt


def _reciprocal(x: int) -> float:
    return 1 / x


class TestMapStreamOnError:
    """Error semantics of the registered map operator."""

    def test_map_update_func_raises_skips_message(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A failing message is logged and dropped; the next one flows."""
        results: list[float] = []
        source = Stream()
        source.map(_reciprocal).sink(results.append)

        with caplog.at_level(
            logging.ERROR,
            logger="functions.streamz_tools",
        ):
            source.emit(0)
        source.emit(1)

        assert results == [1.0]
        assert "dropping message" in caplog.text

    def test_map_update_on_error_raise_propagates(self) -> None:
        """With on_error='raise', the exception stops the stream."""
        source = Stream()
        # Hold a reference: streamz tracks downstream nodes weakly.
        mapped = source.map(_reciprocal, on_error="raise")

        with pytest.raises(ZeroDivisionError):
            source.emit(0)
        assert mapped.upstreams == []

    def test_map_init_invalid_on_error_raises_valueerror(self) -> None:
        """An unsupported on_error value is rejected at wiring time."""
        with pytest.raises(ValueError, match="on_error"):
            Stream().map(_reciprocal, on_error="bogus")


def _result(rc: int, published: bool = True) -> MagicMock:
    info = MagicMock()
    info.rc = rc
    info.is_published.return_value = published
    return info


@pytest.fixture
def mqtt_sink() -> to_mqtt:
    """An MQTT sink wired to a mocked, never-connecting client."""
    sink = to_mqtt(Stream(), host="localhost", port=1883, topic="t")
    sink.client = MagicMock()
    return sink


class TestToMqttPublishRetry:
    """Publish failures reconnect and retry once instead of vanishing."""

    def test_publish_broker_drop_reconnects_and_retries(
        self,
        mqtt_sink: to_mqtt,
    ) -> None:
        """A failed publish reconnects, retries, and confirms delivery."""
        ok = _result(mqtt.MQTT_ERR_SUCCESS)
        mqtt_sink.client.publish.side_effect = [
            _result(mqtt.MQTT_ERR_NO_CONN),
            ok,
        ]

        mqtt_sink._publish("t", b"x")

        mqtt_sink.client.reconnect.assert_called_once()
        assert mqtt_sink.client.publish.call_count == 2
        ok.wait_for_publish.assert_called_once()

    def test_publish_retry_fails_logs_error_and_drops(
        self,
        mqtt_sink: to_mqtt,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A publish failing after reconnect is logged and dropped."""
        second = _result(mqtt.MQTT_ERR_NO_CONN)
        mqtt_sink.client.publish.side_effect = [
            _result(mqtt.MQTT_ERR_NO_CONN),
            second,
        ]

        with caplog.at_level(
            logging.ERROR,
            logger="functions.streamz_tools",
        ):
            mqtt_sink._publish("t", b"x")

        assert "after reconnect" in caplog.text
        second.wait_for_publish.assert_not_called()

    def test_publish_reconnect_raises_logs_and_drops(
        self,
        mqtt_sink: to_mqtt,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An unreachable broker on reconnect does not kill the sink."""
        mqtt_sink.client.publish.return_value = _result(
            mqtt.MQTT_ERR_NO_CONN,
        )
        mqtt_sink.client.reconnect.side_effect = ConnectionRefusedError

        with caplog.at_level(
            logging.ERROR,
            logger="functions.streamz_tools",
        ):
            mqtt_sink._publish("t", b"x")

        assert "reconnect failed" in caplog.text
        mqtt_sink.client.publish.assert_called_once()

    def test_publish_success_no_reconnect(
        self,
        mqtt_sink: to_mqtt,
    ) -> None:
        """A successful publish never touches reconnect."""
        ok = _result(mqtt.MQTT_ERR_SUCCESS)
        mqtt_sink.client.publish.return_value = ok

        mqtt_sink._publish("t", b"x")

        mqtt_sink.client.reconnect.assert_not_called()
        ok.wait_for_publish.assert_called_once()
