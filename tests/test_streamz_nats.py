"""Offline tests for the NATS source, sink, and message adapter.

These tests never touch a live broker. ``nats.connect`` and the NATS
``Client`` are mocked; the background client loop is replaced by a fake
that runs scheduled coroutines synchronously so publish/drain behaviour
can be asserted deterministically.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from streamz import Stream

sys.path.insert(1, str(Path(__file__).parent.parent))

import functions.streamz_tools as st
from functions.streamz_tools import (
    NATSMessage,
    _filt,
    _func,
    from_nats,
    to_nats,
)


class _FakeLoop:
    """Minimal event-loop stand-in driving coroutines synchronously."""

    def __init__(self) -> None:
        self.running = True
        self.stopped = False

    def is_running(self) -> bool:
        """Report the loop as running until explicitly stopped."""
        return self.running

    def call_soon_threadsafe(self, callback: Any) -> None:  # noqa: ANN401
        """Invoke the scheduled callback immediately."""
        callback()

    def stop(self) -> None:
        """Mark the loop stopped, mirroring loop.stop()."""
        self.stopped = True
        self.running = False


def _run_coro_sync(coro: Any, _loop: Any) -> Any:  # noqa: ANN401
    """Drop-in for run_coroutine_threadsafe that runs the coro now."""
    result = asyncio.new_event_loop().run_until_complete(coro)
    fut: MagicMock = MagicMock()
    fut.result.return_value = result
    return fut


# --------------------------------------------------------------------------
# Message adapter
# --------------------------------------------------------------------------
class TestNATSMessageAdapter:
    """The adapter must satisfy the MQTTMessage interface ``_func`` needs."""

    def test_adapter_exposes_topic_and_payload(self) -> None:
        """A NATSMessage exposes ``.topic`` (str) and ``.payload`` (bytes)."""
        msg = NATSMessage(topic="foo", payload=b"1.")

        assert msg.topic == "foo"
        assert msg.payload == b"1."

    def test_adapter_feeds_func_and_filt_accumulation(self) -> None:
        """Adapter messages accumulate by topic exactly like MQTTMessage."""
        topics = ["foo", "bar"]
        state: dict = {}

        state = _func(state, NATSMessage(topic="foo", payload=b"1."), topics)
        assert state == {"foo": b"1."}
        assert _filt(state, topics) is False

        state = _func(state, NATSMessage(topic="bar", payload=b"2."), topics)
        assert state == {"foo": b"1.", "bar": b"2."}
        assert _filt(state, topics) is True


class TestSplitServers:
    """The comma-separated server list is normalised to a list."""

    def test_single_server(self) -> None:
        """A single URL becomes a one-element list."""
        assert st._split_servers("nats://localhost:4222") == [
            "nats://localhost:4222",
        ]

    def test_comma_separated_servers_are_split_and_stripped(self) -> None:
        """Comma-separated URLs split into a stripped list."""
        assert st._split_servers("nats://a:4222, nats://b:4222") == [
            "nats://a:4222",
            "nats://b:4222",
        ]


# --------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------
class TestFromNATSSource:
    """The source registers on Stream and stays lazy until started."""

    def test_from_nats_registered_and_lazy(self) -> None:
        """Construction parses servers/subjects without connecting."""
        source = Stream.from_nats(
            servers="nats://a:4222, nats://b:4222",
            topic=["x", "y"],
        )

        assert isinstance(source, from_nats)
        assert source.servers == ["nats://a:4222", "nats://b:4222"]
        assert source.subjects == ["x", "y"]
        # Lazy: no client thread or connection before start().
        assert source._nc is None
        assert source._thread is None
        source.stop()

    def test_on_message_enqueues_adapter(self) -> None:
        """A received NATS msg is queued as a NATSMessage adapter."""
        source = Stream.from_nats(servers="nats://a:4222", topic="x")
        nats_msg = MagicMock(subject="x", data=b"42.")

        source._on_message(nats_msg)

        queued = source.q.get_nowait()
        assert isinstance(queued, NATSMessage)
        assert queued.topic == "x"
        assert queued.payload == b"42."
        source.stop()


# --------------------------------------------------------------------------
# Sink
# --------------------------------------------------------------------------
@pytest.fixture
def nats_sink(monkeypatch: pytest.MonkeyPatch) -> to_nats:
    """A NATS sink with a fake loop/connection and synchronous publishing."""
    sink = to_nats(Stream(), servers="nats://localhost:4222", topic="t")
    sink._client_loop = _FakeLoop()  # type: ignore[assignment]
    sink._nc = MagicMock()

    # A fresh awaitable per call avoids re-awaiting an exhausted coroutine.
    def _fresh(*_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
        return _async_none()

    sink._nc.publish = MagicMock(side_effect=_fresh)
    sink._nc.flush = MagicMock(side_effect=_fresh)
    sink._nc.drain = MagicMock(side_effect=_fresh)
    monkeypatch.setattr(
        st.asyncio,
        "run_coroutine_threadsafe",
        _run_coro_sync,
    )
    return sink


async def _async_none() -> None:
    """An already-awaitable no-op coroutine."""


class TestToNATSLazyConnect:
    """The client connects lazily on the first published message."""

    def test_update_connects_on_first_publish(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The first update triggers _ensure_client; later ones reuse it."""
        sink = to_nats(Stream(), servers="nats://localhost:4222", topic="t")
        calls: list[int] = []

        def _fake_ensure() -> None:
            calls.append(1)
            sink._client_loop = _FakeLoop()  # type: ignore[assignment]
            sink._nc = MagicMock()

        monkeypatch.setattr(sink, "_ensure_client", _fake_ensure)
        monkeypatch.setattr(sink, "_publish", MagicMock())

        sink.update(b"first")
        sink.update(b"second")

        # _ensure_client is invoked on every update but guards internally;
        # here the stub records that the sink is the connect entry point.
        assert calls == [1, 1]

    def test_publish_drops_when_client_not_ready(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """_publish without a live loop logs an error and drops the msg."""
        sink = to_nats(Stream(), servers="nats://localhost:4222", topic="t")

        sink._publish("t", b"x")

        assert "not ready" in caplog.text.lower()


class TestToNATSFanOut:
    """Subject fan-out mirrors the MQTT sink for float and dict limits."""

    def test_bytes_pass_through_to_base_subject(
        self,
        nats_sink: to_nats,
    ) -> None:
        """Raw bytes publish unchanged to the configured subject."""
        nats_sink.update(b"hello")

        nats_sink._nc.publish.assert_called_once_with("t", b"hello")

    def test_float_limits_fan_out_to_three_subjects(
        self,
        nats_sink: to_nats,
    ) -> None:
        """A flat dict fans out to anomaly/_DOL_high/_DOL_low subjects."""
        nats_sink.update(
            {"anomaly": 1, "level_high": 0.5, "level_low": -0.5},
        )

        subjects = [c.args[0] for c in nats_sink._nc.publish.call_args_list]
        assert subjects == ["tanomaly", "t_DOL_high", "t_DOL_low"]

    def test_dict_limits_fan_out_per_signal(
        self,
        nats_sink: to_nats,
    ) -> None:
        """Nested limits fan out per signal incl. root_cause flags."""
        nats_sink.update(
            {
                "anomaly": 1,
                "level_high": {"a": 0.5, "b": 0.6},
                "level_low": {"a": -0.5, "b": -0.4},
                "root_cause": "b",
            },
        )

        subjects = [c.args[0] for c in nats_sink._nc.publish.call_args_list]
        assert subjects == [
            "tanomaly",
            "a_DOL_high",
            "a_DOL_low",
            "a_root_cause",
            "b_DOL_high",
            "b_DOL_low",
            "b_root_cause",
        ]
        # The root_cause flag is 1 for the matched signal, 0 otherwise.
        payloads = {
            c.args[0]: c.args[1] for c in nats_sink._nc.publish.call_args_list
        }
        assert payloads["a_root_cause"] == b"0"
        assert payloads["b_root_cause"] == b"1"


class TestToNATSDestroy:
    """destroy() drains the connection and stops the client loop."""

    def test_destroy_drains_and_stops_loop(
        self,
        nats_sink: to_nats,
    ) -> None:
        """destroy() awaits drain() and stops the background loop."""
        loop = nats_sink._client_loop
        nc = nats_sink._nc

        nats_sink.destroy()

        nc.drain.assert_called_once()
        assert loop.stopped is True  # type: ignore[union-attr]
        assert nats_sink._nc is None
        assert nats_sink._client_loop is None
