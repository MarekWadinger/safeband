"""Streamz stream operators and MQTT sink for real-time data pipelines."""

import logging
from collections.abc import Callable

import paho.mqtt.client as mqtt
from paho.mqtt.client import MQTTMessage
from streamz import Sink, Stream

logger = logging.getLogger(__name__)


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
                    self._publish(f"{key}_DOL_high", x["level_high"][key])
                    self._publish(f"{key}_DOL_low", x["level_low"][key])
                    self._publish(
                        f"{key}_root_cause",
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


def _func(previous_state: dict, new_msg: MQTTMessage, topics: list) -> dict:
    """Update the state with the new message.

    Args:
        previous_state (dict): Dictionary of previous messages.
        new_msg (MQTTMessage): New message.
        topics (list): List of required topics.

    Returns:
        dict: Updated state.

    Examples:
    >>> previous_state = {}
    >>> topics = ['foo']
    >>> new_msg = MQTTMessage(topic=b'foo')
    >>> new_msg.payload = b'1.'
    >>> previous_state = _func(previous_state, new_msg, topics)
    >>> previous_state
    {'foo': b'1.'}
    >>> new_msg = MQTTMessage(topic=b'bar')
    >>> new_msg.payload = b'1.'
    >>> previous_state = _func(previous_state, new_msg, topics)
    >>> previous_state
    {'foo': b'1.'}
    >>> new_msg = MQTTMessage(topic=b'foo')
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
