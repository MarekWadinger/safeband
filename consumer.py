import datetime as dt
import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, cast

import paho.mqtt.client as mqtt

from functions.encryption import init_rsa_security, verify_and_decrypt_data
from functions.parse import get_params
from functions.typing_extras import FileClient, MQTTClient, istypedinstance

logger = logging.getLogger(__name__)

PORT = 1883


# MQTT callback functions
def on_connect(self: mqtt.Client, userdata, _flags, rc) -> None:
    """MQTT callback function for handling the connect event.

    Args:
        userdata: User-specific data passed to the callback.
        flags: Response flags from the broker.
        rc: The connection result code.

    Examples:
        >>> obj = mqtt.Client()
        >>> usr = Namespace(topic=["my_topic"])
        >>> on_connect(mqtt.Client(), usr, None, 0)

    """
    logger.info("Connected with result code %s", rc)
    self.subscribe([(topic, 0) for topic in userdata.topic])


def on_message(_self, userdata, msg) -> None:
    """MQTT callback function for handling incoming messages.

    Args:
        userdata: User-specific data passed to the callback.
        msg: The message received from the broker.

    Examples:
        >>> obj = mqtt.Client()
        >>> usr = Namespace(topic=["my_topic"])
        >>> msg = mqtt.MQTTMessage(); msg.payload = b'Hello'
        >>> on_message(obj, usr, msg)

    """
    if isinstance(userdata, Namespace) and "receiver" in userdata:
        item = verify_and_decrypt_data(
            json.loads(msg.payload.decode()),
            userdata.receiver,
        )
        item = json.dumps(item)
    else:
        item = msg.payload.decode()
    t = dt.datetime.fromtimestamp(msg.timestamp, tz=dt.UTC).replace(
        microsecond=0
    )
    logger.info("Received message at %s: %s", t, item)


def query_file(config: FileClient, **kwargs) -> None:
    """Query a JSON file based on the command-line arguments and print the
    closest past item.


    Args:
        config (dict): The configuration dictionary.
        args (Namespace): Parsed command-line arguments.

    Examples:
        >>> config = {"output": "tests/sample.json"}
        >>> query_file(config)

    """
    # Load the JSON file as a list of dictionaries
    with Path(config.get("output", "")).open(encoding="utf-8") as f:
        data: list[dict[str, Any]] = [json.loads(line) for line in f]

    # Convert the time strings to datetime objects
    for i, item in enumerate(data):
        if "receiver" in kwargs and not item["time"].isascii():
            data[i] = cast(
                "dict[str, Any]",
                verify_and_decrypt_data(item, kwargs["receiver"]),
            )
        data[i]["time"] = dt.datetime.strptime(
            str(data[i]["time"]),
            "%Y-%m-%d %H:%M:%S",
        ).replace(tzinfo=dt.UTC)

    # Sort the data by time in descending order
    data.sort(key=lambda x: x["time"], reverse=True)

    # Find the closest past item
    closest_item = None
    for item in data:
        if item["time"] <= dt.datetime.now(dt.UTC).replace(microsecond=0):
            closest_item = item
            break

    logger.info("%s", closest_item)


def query_mqtt(config: MQTTClient):
    """Create an MQTT client instance and connect to the MQTT broker.

    Args:
        config (dict): The configuration dictionary.
        args (Namespace): Parsed command-line arguments.

    Returns:
        mqtt.Client: MQTT client instance.

    Examples:
        >>> client = mqtt.Client()
        >>> isinstance(client, mqtt.Client)
        True

    """
    # Create MQTT client instance
    client = mqtt.Client()

    # Assign callback functions
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect to the MQTT broker
    client.connect(config["host"], PORT, 60)
    return client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = get_params()

    if "key_path" in config["setup"]:
        _, receiver = init_rsa_security(config["setup"]["key_path"])

    client = config["client"]
    if istypedinstance(client, FileClient):
        query_file(cast("FileClient", client), receiver=receiver)
    elif istypedinstance(client, MQTTClient):
        client = query_mqtt(cast("MQTTClient", client))
        client.loop_forever()
