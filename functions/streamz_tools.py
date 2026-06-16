"""Streamz stream operators and MQTT/NATS sinks for real-time pipelines."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import nats
import paho.mqtt.client as mqtt
from streamz import Sink, Stream
from streamz.sources import from_q

if TYPE_CHECKING:
    from collections.abc import Callable

    from nats.aio.client import Client as NATSConnection
    from nats.aio.msg import Msg as NATSMsg

logger = logging.getLogger(__name__)

# Characters that carry routing meaning in MQTT topics or NATS subjects:
# the MQTT single/multi-level wildcards, the MQTT level separator, and the
# NATS token separator. A feature name must not smuggle any of these into a
# published subject (topic injection / unintended fan-out).
_UNSAFE_SUBJECT_CHARS = frozenset({"+", "#", "/", "."})


def _safe_subject_token(key: str) -> str:
    """Return ``key`` unchanged or reject it as an unsafe subject token.

    A feature name is interpolated into MQTT topics and NATS subjects, so
    it must not contain characters with routing meaning: the MQTT
    wildcards, the MQTT level separator, or the NATS token separator.
    Such a name could subscribe-match unintended topics or split into
    extra subject levels.

    Args:
        key: The feature name to validate.

    Returns:
        str: The validated token, unchanged.

    Raises:
        ValueError: If ``key`` contains a wildcard or level separator.

    """
    if _UNSAFE_SUBJECT_CHARS.intersection(key):
        msg = (
            f"Unsafe subject token {key!r}: feature names must not contain "
            "MQTT/NATS wildcards or level separators."
        )
        raise ValueError(msg)
    return key


class TopicMessage(Protocol):
    """Structural type for messages keyed by topic with a byte payload.

    Both paho's ``MQTTMessage`` and the :class:`NATSMessage` adapter
    satisfy this protocol, so ``_func``/``_filt`` accept either transport's
    messages without depending on a concrete class. The members are
    declared read-only (as properties) so concrete ``str``/``bytes``
    topics remain assignable under protocol covariance.
    """

    @property
    def topic(self) -> str | bytes:
        """The subject/topic the message arrived on."""
        ...

    @property
    def payload(self) -> bytes:
        """The raw message body."""
        ...


def _split_servers(servers: str | list[str]) -> list[str]:
    """Normalise a NATS ``servers`` spec into a list of URLs.

    Args:
        servers (str | list[str]): A single URL, a comma-separated string
            of URLs, or an already-split list of URLs.

    Returns:
        list[str]: The individual, whitespace-stripped server URLs.

    Examples:
    >>> _split_servers("nats://localhost:4222")
    ['nats://localhost:4222']
    >>> _split_servers("nats://a:4222, nats://b:4222")
    ['nats://a:4222', 'nats://b:4222']
    >>> _split_servers(["nats://a:4222"])
    ['nats://a:4222']

    """
    if isinstance(servers, list):
        return [s.strip() for s in servers if s.strip()]
    return [s.strip() for s in servers.split(",") if s.strip()]


@dataclass
class NATSMessage:
    """Adapter exposing a NATS message with the MQTT message interface.

    The existing ``_func``/``_filt`` accumulator and ``preprocess`` read
    ``msg.topic`` (str) and ``msg.payload`` (bytes), modelled on paho's
    ``MQTTMessage``. NATS delivers ``subject``/``data`` instead, so the
    source wraps every received message in this adapter to keep the
    downstream pipeline unchanged.

    Args:
        topic (str): The NATS subject the message arrived on.
        payload (bytes): The raw message body.

    Examples:
    >>> m = NATSMessage(topic="foo", payload=b"1.")
    >>> m.topic, m.payload
    ('foo', b'1.')

    """

    topic: str
    payload: bytes


@Stream.register_api(attribute_name="map")
class MapStream(Stream):
    """Stream operator that applies a function to each upstream element.

    Args:
        upstream (Stream): Upstream stream.
        func (Callable): Function applied to each element.
        *args: Extra positional arguments passed to ``func``.
        on_error (str): Either ``"skip"`` (default) to log and drop a
            message whose processing raised, or ``"raise"`` to stop the
            stream and re-raise the exception.
        **kwargs: Extra keyword arguments passed to ``func``.
    """

    def __init__(
        self,
        upstream: Stream | None,
        func: Callable,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Store func and extra args, then initialize the parent Stream."""
        self.func = func
        # this is one of a few stream specific kwargs
        stream_name = kwargs.pop("stream_name", None)
        on_error = kwargs.pop("on_error", "skip")
        if on_error not in ("skip", "raise"):
            msg = f"on_error must be 'skip' or 'raise', got {on_error!r}"
            raise ValueError(msg)
        self.on_error = on_error
        self.kwargs = kwargs
        self.args = args

        Stream.__init__(self, upstream, stream_name=stream_name)

    def update(
        self,
        x: object,
        who: Stream | None = None,
        metadata: list | None = None,
    ) -> object:
        """Apply func to x and emit the result, handling errors per on_error.

        With ``on_error="skip"`` a failing message is logged and dropped so
        one bad payload cannot take down the service; with ``"raise"`` the
        stream is stopped and the exception propagates.
        """
        del who  # unused; required by the streamz Stream.update API
        try:
            result = self.func(x, *self.args, **self.kwargs)
        except Exception:
            if self.on_error == "raise":
                self.stop()
                self.destroy()
                logger.exception("Stream update failed")
                raise
            logger.exception("Stream update failed; dropping message")
            return None
        else:
            return self._emit(result, metadata=metadata)


@Stream.register_api()
class to_mqtt(Sink):
    """Streamz Sink that publishes upstream messages to an MQTT broker.

    Args:
        upstream (Stream): Upstream stream.
        host (str): MQTT broker host.
        port (int): MQTT broker port.
        topic (str): MQTT topic.
        keepalive (int): Keepalive duration.
        client_kwargs (dict): Additional arguments for MQTT client connect.
        publish_kwargs (dict): Additional arguments for MQTT publish.
        publish_timeout (float): Seconds to wait for delivery confirmation
            of each published message before logging a warning.
        **kwargs: Additional keyword arguments.

    Examples:
    The publish/subscribe lines below talk to the public broker
    ``test.mosquitto.org`` and are marked ``# doctest: +SKIP``: the
    round-trip is non-deterministic (a shared public topic) and
    ``subscribe.simple`` blocks with no timeout, so running them under
    CI both flakes and hangs. Construction is lazy and stays offline.

    >>> import datetime as dt
    >>> out_msg = bytes(str(dt.datetime.utcnow()), encoding='utf-8')
    >>> mqtt_sink = to_mqtt(
    ...     Stream(), host="test.mosquitto.org",
    ...     port=1883, topic='adaptive-interpretable-ad/test',
    ...     publish_kwargs={"retain":True})
    >>> mqtt_sink.update(out_msg)  # doctest: +SKIP

    Check the message
    >>> import paho.mqtt.subscribe as subscribe
    >>> msg = subscribe.simple(hostname="test.mosquitto.org",  # doctest: +SKIP
    ...                        topics="adaptive-interpretable-ad/test")
    >>> msg.payload == out_msg  # doctest: +SKIP
    True

    Publish a dictionary
    >>> out_msg = {
    ...     'anomaly': 1,
    ...     'level_high': 0.5,
    ...     'level_low': -0.5,
    ...     }
    >>> mqtt_sink.update(out_msg)  # doctest: +SKIP

    Check the message
    >>> import paho.mqtt.subscribe as subscribe
    >>> msg = subscribe.simple(hostname="test.mosquitto.org",  # doctest: +SKIP
    ...                        topics="adaptive-interpretable-ad/testanomaly")
    >>> int(msg.payload) == out_msg['anomaly']  # doctest: +SKIP
    True

    Publish a nested dictionary
    >>> out_msg = {
    ...     'anomaly': 1,
    ...     'level_high': {'a': 0.5, 'b': 0.6},
    ...     'level_low': {'a': -0.5, 'b': -0.4},
    ...     'root_cause': 'b',
    ...     }
    >>> mqtt_sink.update(out_msg)  # doctest: +SKIP

    Check the message
    >>> import paho.mqtt.subscribe as subscribe
    >>> msg = subscribe.simple(hostname="test.mosquitto.org",  # doctest: +SKIP
    ...                        topics="b_DOL_high")
    >>> float(msg.payload) == out_msg['level_high']['b']  # doctest: +SKIP
    True

    >>> mqtt_sink.destroy()  # doctest: +SKIP

    """

    def __init__(
        self,
        upstream: Stream,
        host: str,
        port: int,
        topic: str,
        keepalive: int = 60,
        client_kwargs: dict | None = None,
        publish_kwargs: dict | None = None,
        publish_timeout: float = 5.0,
        **kwargs: object,
    ) -> None:
        """Store connection parameters and initialize the Sink."""
        self.host = host
        self.port = port
        self.c_kw = client_kwargs or {}
        self.p_kw = publish_kwargs or {}
        self.client: mqtt.Client | None = None
        self.topic = topic
        self.keepalive = keepalive
        self.publish_timeout = publish_timeout
        super().__init__(upstream, ensure_io_loop=True, **kwargs)

    def _publish(self, topic: str, payload: object) -> None:
        """Publish one message and wait for its delivery confirmation.

        paho reports failures (e.g. a dropped broker connection) via the
        result code instead of raising, so a failed publish reconnects
        and retries once before giving up with an error log.
        """
        assert self.client is not None  # narrowed by update()
        info = self.client.publish(topic, payload, **self.p_kw)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.warning(
                "MQTT publish failed (rc=%s) for topic %r; "
                "reconnecting and retrying once",
                info.rc,
                topic,
            )
            try:
                self.client.reconnect()
            except OSError:
                logger.exception("MQTT reconnect failed for topic %r", topic)
                return
            info = self.client.publish(topic, payload, **self.p_kw)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.error(
                    "MQTT publish failed after reconnect (rc=%s) "
                    "for topic %r; message dropped",
                    info.rc,
                    topic,
                )
                return
        info.wait_for_publish(timeout=self.publish_timeout)
        if not info.is_published():
            logger.warning(
                "MQTT delivery not confirmed within %.1fs for topic %r",
                self.publish_timeout,
                topic,
            )

    def update(
        self,
        x: bytes | dict,
        who: Stream | None = None,
        metadata: list | None = None,
    ) -> None:
        """Publish x to the MQTT broker, connecting lazily on first call."""
        del who, metadata  # unused; required by the streamz Sink.update API

        if self.client is None:
            self.client = mqtt.Client(clean_session=True)
            self.client.connect(
                self.host,
                self.port,
                self.keepalive,
                **self.c_kw,
            )
            # Run the network loop in the background so the broker
            # handshake completes and wait_for_publish can confirm
            # delivery.
            self.client.loop_start()
        if isinstance(x, bytes):
            self._publish(self.topic, x)
        else:
            self._publish(f"{self.topic}anomaly", x["anomaly"])
            if isinstance(x["level_high"], dict):
                for key in x["level_high"]:
                    safe_key = _safe_subject_token(key)
                    self._publish(f"{safe_key}_DOL_high", x["level_high"][key])
                    self._publish(f"{safe_key}_DOL_low", x["level_low"][key])
                    self._publish(
                        f"{safe_key}_root_cause",
                        1 if key == x["root_cause"] else 0,
                    )
            else:
                self._publish(f"{self.topic}_DOL_high", x["level_high"])
                self._publish(f"{self.topic}_DOL_low", x["level_low"])

    def destroy(self) -> None:
        """Disconnect the MQTT client and destroy the sink."""
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
            self.client = None
            super().destroy()


@Stream.register_api(staticmethod)
class from_nats(from_q):
    """Streamz Source that reads messages from one or more NATS subjects.

    streamz ships no built-in NATS source, so this subclass bridges the
    asyncio-only ``nats-py`` client into streamz's IOLoop. nats-py is not
    safe to drive from a foreign event loop, so the client runs on a
    dedicated background thread with its own asyncio loop; its message
    callback pushes :class:`NATSMessage` adapters into a thread-safe
    ``queue.Queue``. The inherited ``from_q`` polling loop, running on
    streamz's own IOLoop, drains that queue and emits into the pipeline.
    This mirrors how the built-in ``from_mqtt`` bridges paho's background
    network thread.

    Each emitted item exposes ``.topic`` (the NATS subject, ``str``) and
    ``.payload`` (``bytes``) so the existing ``_func``/``_filt``
    accumulator and ``preprocess`` work unchanged.

    Args:
        servers (str | list[str]): NATS URL(s); a comma-separated string
            is accepted and split into a list.
        topic (str | list[str]): Subject or list of subjects to subscribe
            to.
        connect_kwargs (dict | None): Extra arguments forwarded to
            ``nats.connect``.
        sleep_time (float): Seconds the polling loop waits when the queue
            is empty.
        **kwargs: Forwarded to ``streamz.Source``.

    Examples:
    Construction is lazy; the client only connects once the source is
    started, so wiring the node stays offline.

    >>> from streamz import Stream
    >>> source = Stream.from_nats(
    ...     servers="nats://localhost:4222",
    ...     topic="my_subject",
    ... )
    >>> source.subjects
    ['my_subject']
    >>> source.servers
    ['nats://localhost:4222']
    >>> source.stop()

    """

    def __init__(
        self,
        servers: str | list[str],
        topic: str | list[str],
        connect_kwargs: dict | None = None,
        sleep_time: float = 0.01,
        **kwargs: object,
    ) -> None:
        """Store connection parameters and initialise the polling source."""
        self.servers = _split_servers(servers)
        self.subjects = [topic] if isinstance(topic, str) else list(topic)
        self.connect_kwargs = connect_kwargs or {}
        self._nc: NATSConnection | None = None
        self._thread: threading.Thread | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        super().__init__(
            q=queue.Queue(),
            sleep_time=sleep_time,
            **kwargs,
        )

    def _on_message(self, msg: NATSMsg) -> None:
        """Queue a received NATS message as an MQTT-compatible adapter."""
        self.q.put(
            NATSMessage(topic=msg.subject, payload=msg.data),
        )

    async def _connect_and_subscribe(self) -> None:
        """Connect to NATS and subscribe to every configured subject."""
        nc = await nats.connect(
            servers=self.servers,
            **self.connect_kwargs,
        )
        self._nc = nc
        for subject in self.subjects:
            await nc.subscribe(subject, cb=self._on_message_async)

    async def _on_message_async(self, msg: NATSMsg) -> None:
        """Async subscription callback delegating to the queue push."""
        self._on_message(msg)

    def _run_client_loop(self) -> None:
        """Drive the nats-py client on its own asyncio event loop."""
        loop = asyncio.new_event_loop()
        self._client_loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._connect_and_subscribe())
            loop.run_forever()
        finally:
            loop.close()

    def start(self) -> None:
        """Start the background NATS client and the polling loop."""
        if self.stopped:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run_client_loop,
                    daemon=True,
                    name="from_nats-client",
                )
                self._thread.start()
            super().start()

    def stop(self) -> None:
        """Stop polling and tear down the background NATS client."""
        super().stop()
        loop = self._client_loop
        nc = self._nc
        if loop is not None and loop.is_running():

            async def _close() -> None:
                if nc is not None:
                    await nc.drain()

            try:
                fut = asyncio.run_coroutine_threadsafe(_close(), loop)
                fut.result(timeout=5)
            except Exception:  # noqa: BLE001
                logger.warning("NATS drain on stop failed", exc_info=True)
            loop.call_soon_threadsafe(loop.stop)
        self._nc = None
        self._client_loop = None


@Stream.register_api()
class to_nats(Sink):
    """Streamz Sink that publishes upstream messages to a NATS server.

    nats-py is asyncio-only, so the client runs on a dedicated background
    thread with its own event loop. Publishes are scheduled onto that loop
    with ``run_coroutine_threadsafe`` and awaited for back-pressure. The
    payload fan-out mirrors :class:`to_mqtt`: raw bytes pass through to the
    configured subject, while dict results are split into ``anomaly``,
    ``*_DOL_high``, ``*_DOL_low`` and ``*_root_cause`` subjects.

    Args:
        upstream (Stream): Upstream stream.
        servers (str | list[str]): NATS URL(s); a comma-separated string
            is accepted and split into a list.
        topic (str): Subject prefix for published messages.
        connect_kwargs (dict | None): Extra arguments for ``nats.connect``.
        publish_timeout (float): Seconds to wait for each publish to be
            scheduled and flushed before logging a warning.
        **kwargs: Additional keyword arguments for the Sink.

    Examples:
    Construction is lazy and stays offline; the live publish/round-trip is
    marked ``# doctest: +SKIP`` because it needs a running NATS server.

    >>> from streamz import Stream
    >>> nats_sink = to_nats(
    ...     Stream(),
    ...     servers="nats://localhost:4222",
    ...     topic="adaptive-interpretable-ad/test",
    ... )
    >>> nats_sink.servers
    ['nats://localhost:4222']
    >>> nats_sink.update(b"hello")  # doctest: +SKIP

    Publish a dictionary of dynamic limits
    >>> out_msg = {
    ...     'anomaly': 1,
    ...     'level_high': 0.5,
    ...     'level_low': -0.5,
    ...     }
    >>> nats_sink.update(out_msg)  # doctest: +SKIP
    >>> nats_sink.destroy()  # doctest: +SKIP

    """

    def __init__(
        self,
        upstream: Stream,
        servers: str | list[str],
        topic: str,
        connect_kwargs: dict | None = None,
        publish_timeout: float = 5.0,
        **kwargs: object,
    ) -> None:
        """Store connection parameters and initialise the Sink."""
        self.servers = _split_servers(servers)
        self.topic = topic
        self.connect_kwargs = connect_kwargs or {}
        self.publish_timeout = publish_timeout
        self._nc: NATSConnection | None = None
        self._thread: threading.Thread | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        super().__init__(upstream, ensure_io_loop=True, **kwargs)

    def _ensure_client(self) -> None:
        """Connect lazily, starting the client loop thread on first use."""
        if self._nc is not None:
            return
        ready: Future[None] = Future()

        async def _connect() -> None:
            self._nc = await nats.connect(
                servers=self.servers,
                **self.connect_kwargs,
            )

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            self._client_loop = loop
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_connect())
                ready.set_result(None)
            except Exception as exc:  # noqa: BLE001
                ready.set_exception(exc)
                return
            loop.run_forever()

        self._thread = threading.Thread(
            target=_run_loop,
            daemon=True,
            name="to_nats-client",
        )
        self._thread.start()
        ready.result(timeout=self.publish_timeout)

    def _publish(self, subject: str, payload: object) -> None:
        """Publish one message on ``subject`` and flush on the client loop."""
        loop = self._client_loop
        nc = self._nc
        if loop is None or nc is None:  # pragma: no cover - guarded by update
            logger.error("NATS client not ready; dropping %r", subject)
            return
        data = payload if isinstance(payload, bytes) else str(payload).encode()

        async def _do() -> None:
            await nc.publish(subject, data)
            await nc.flush()

        try:
            fut = asyncio.run_coroutine_threadsafe(_do(), loop)
            fut.result(timeout=self.publish_timeout)
        except Exception:  # noqa: BLE001
            logger.warning(
                "NATS publish to %r failed; message dropped",
                subject,
                exc_info=True,
            )

    def update(
        self,
        x: bytes | dict,
        who: Stream | None = None,
        metadata: list | None = None,
    ) -> None:
        """Publish x to NATS, connecting lazily on first call."""
        del who, metadata  # unused; required by the streamz Sink.update API
        self._ensure_client()
        if isinstance(x, bytes):
            self._publish(self.topic, x)
        else:
            self._publish(f"{self.topic}anomaly", x["anomaly"])
            if isinstance(x["level_high"], dict):
                for key in x["level_high"]:
                    safe_key = _safe_subject_token(key)
                    self._publish(f"{safe_key}_DOL_high", x["level_high"][key])
                    self._publish(f"{safe_key}_DOL_low", x["level_low"][key])
                    self._publish(
                        f"{safe_key}_root_cause",
                        1 if key == x["root_cause"] else 0,
                    )
            else:
                self._publish(f"{self.topic}_DOL_high", x["level_high"])
                self._publish(f"{self.topic}_DOL_low", x["level_low"])

    def destroy(self) -> None:
        """Drain and close the NATS connection, then destroy the sink."""
        loop = self._client_loop
        nc = self._nc
        if loop is not None and nc is not None:

            async def _close() -> None:
                await nc.drain()

            try:
                fut = asyncio.run_coroutine_threadsafe(_close(), loop)
                fut.result(timeout=self.publish_timeout)
            except Exception:  # noqa: BLE001
                logger.warning("NATS drain on destroy failed", exc_info=True)
            if loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
        self._nc = None
        self._client_loop = None
        super().destroy()


def _filt(msgs: dict, topics: list) -> bool:
    """Check availability of all topics in the dictionary.

    Args:
        msgs (dict): Dictionary of messages.
        topics (list): List of topics checked for availability in msgs

    Returns:
        bool: True if all topics are available in msgs, False otherwise.

    Examples:
    >>> msgs = {'a': 1, 'b': 2}
    >>> topics = ['a', 'b']
    >>> _filt(msgs, topics)
    True
    >>> topics = ['a', 'b', 'c']
    >>> _filt(msgs, topics)
    False

    """
    return all(topic in msgs for topic in topics)


def _func(previous_state: dict, new_msg: TopicMessage, topics: list) -> dict:
    """Update the state with the new message.

    Args:
        previous_state (dict): Dictionary of previous messages.
        new_msg (TopicMessage): New message exposing ``topic`` and
            ``payload`` (an MQTTMessage or a NATSMessage adapter).
        topics (list): List of required topics.

    Returns:
        dict: Updated state.

    Examples:
    >>> previous_state = {}
    >>> topics = ['foo']
    >>> new_msg = mqtt.MQTTMessage(topic=b'foo')
    >>> new_msg.payload = b'1.'
    >>> previous_state = _func(previous_state, new_msg, topics)
    >>> previous_state
    {'foo': b'1.'}
    >>> new_msg = mqtt.MQTTMessage(topic=b'bar')
    >>> new_msg.payload = b'1.'
    >>> previous_state = _func(previous_state, new_msg, topics)
    >>> previous_state
    {'foo': b'1.'}
    >>> new_msg = mqtt.MQTTMessage(topic=b'foo')
    >>> new_msg.payload = b'2.'
    >>> _func(previous_state, new_msg, topics)
    {'foo': b'2.'}

    """
    if new_msg.topic in topics:
        if not _filt(previous_state, topics):
            previous_state[new_msg.topic] = new_msg.payload
            state = previous_state.copy()
        else:
            state = {new_msg.topic: new_msg.payload}
    else:
        state = previous_state.copy()
    return state
