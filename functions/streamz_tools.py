import logging
from typing import Union

import paho.mqtt.client as mqtt
from streamz import Sink, Stream
from paho.mqtt.client import MQTTMessage

logger = logging.getLogger(__name__)


@Stream.register_api()
class map(Stream):
    def __init__(self, upstream, func, *args, **kwargs):
        self.func = func
        # this is one of a few stream specific kwargs
        stream_name = kwargs.pop('stream_name', None)
        self.kwargs = kwargs
        self.args = args

        Stream.__init__(self, upstream, stream_name=stream_name)

    def update(self, x, who=None, metadata=None):
        try:
            result = self.func(x, *self.args, **self.kwargs)
        except Exception as e:
            self.stop()
            self.destroy()
            logger.exception(e)
            raise e
        else:
            return self._emit(result, metadata=metadata)


@Stream.register_api()
class to_mqtt(Sink):
    """
    Initialize the to_mqtt instance.

    Args:
        upstream (Stream): Upstream stream.
        host (str): MQTT broker host.
        port (int): MQTT broker port.
        topic (str): MQTT topic.
        keepalive (int): Keepalive duration.
        client_kwargs (dict): Additional arguments for MQTT client connect.
        publish_kwargs (dict): Additional arguments for MQTT publish.
        **kwargs: Additional keyword arguments.

    Examples:
    >>> import datetime as dt
    >>> out_msg = bytes(str(dt.datetime.utcnow()), encoding='utf-8')
    >>> mqtt_sink = to_mqtt(
    ...     Stream(), host="mqtt.eclipseprojects.io",
    ...     port=1883, topic='test', publish_kwargs={"retain":True})
    >>> mqtt_sink.update(out_msg)

    Check the message
    >>> import paho.mqtt.subscribe as subscribe
    >>> msg = subscribe.simple(hostname="mqtt.eclipseprojects.io",
    ...                        topics="test")
    >>> msg.payload == out_msg
    True

    Publish a dictionary
    >>> out_msg = {
    ...     'anomaly': 1,
    ...     'level_high': 0.5,
    ...     'level_low': -0.5,
    ...     }
    >>> mqtt_sink.update(out_msg)

    Check the message
    >>> import paho.mqtt.subscribe as subscribe
    >>> msg = subscribe.simple(hostname="mqtt.eclipseprojects.io",
    ...                        topics="testanomaly")
    >>> int(msg.payload) == out_msg['anomaly']
    True

    Publish a nested dictionary
    >>> out_msg = {
    ...     'anomaly': 1,
    ...     'level_high': {'a': 0.5, 'b': 0.6},
    ...     'level_low': {'a': -0.5, 'b': -0.4},
    ...     }
    >>> mqtt_sink.update(out_msg)

    Check the message
    >>> import paho.mqtt.subscribe as subscribe
    >>> msg = subscribe.simple(hostname="mqtt.eclipseprojects.io",
    ...                        topics="b_DOL_high")
    >>> float(msg.payload) == out_msg['level_high']['b']
    True

    >>> mqtt_sink.destroy()
    """
    def __init__(self, upstream, host, port, topic, keepalive=60,
                 client_kwargs=None, publish_kwargs=None,
                 **kwargs):
        self.host = host
        self.port = port
        self.c_kw = client_kwargs or {}
        self.p_kw = publish_kwargs or {}
        self.client: Union[mqtt.Client, None] = None
        self.topic = topic
        self.keepalive = keepalive
        super().__init__(upstream, ensure_io_loop=True, **kwargs)

    def update(self, x, who=None, metadata=None):
        if self.client is None:
            self.client = mqtt.Client(clean_session=True)
            self.client.connect(self.host, self.port, self.keepalive,
                                **self.c_kw)
        # TODO: wait on successful delivery
        if isinstance(x, bytes):
            self.client.publish(self.topic, x, **self.p_kw)
        else:
            self.client.publish(
                f"{self.topic}anomaly", x['anomaly'], **self.p_kw)
            if isinstance(x['level_high'], dict):
                for key in x['level_high']:
                    self.client.publish(
                        f"{key}_DOL_high", x['level_high'][key], **self.p_kw)
                    self.client.publish(
                        f"{key}_DOL_low", x['level_low'][key], **self.p_kw)
            else:
                self.client.publish(
                    f"{self.topic}_DOL_high", x['level_high'], **self.p_kw)
                self.client.publish(
                    f"{self.topic}_DOL_low", x['level_low'], **self.p_kw)

    def destroy(self):
        if self.client is not None:
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
    MQTTMessage()
    if new_msg.topic in topics:
        if not _filt(previous_state, topics):
            previous_state[new_msg.topic] = new_msg.payload
            state = previous_state.copy()
        else:
            state = {new_msg.topic: new_msg.payload}
    else:
        state = previous_state.copy()
    return state
