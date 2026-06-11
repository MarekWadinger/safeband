"""MQTT and file-based consumer for anomaly detection results."""

import datetime as dt
import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, cast

import paho.mqtt.client as mqtt
from human_security import HumanRSA

from functions.encryption import init_rsa_security, verify_and_decrypt_data
from functions.parse import get_params
from functions.typing_extras import FileClient, MQTTClient, istypedinstance

logger = logging.getLogger(__name__)

PORT = 1883


# MQTT callback functions
def on_connect(
    self: mqtt.Client,
    userdata: Namespace,
    _flags: dict[str, int],
    rc: int,
) -> None:
    """Subscribe to configured topics after a successful broker connection.

    Args:
        self: MQTT client instance invoking the callback.
        userdata: User-specific data passed to the callback.
        _flags: Response flags from the broker (unused).
        rc: The connection result code.

    """
    logger.info("Connected with result code %s", rc)
    self.subscribe([(topic, 0) for topic in userdata.topic])


def on_message(
    _self: mqtt.Client,
    userdata: Namespace | None,
    msg: mqtt.MQTTMessage,
) -> None:
    """Decrypt and log an incoming MQTT message.

    Args:
        _self: MQTT client instance (unused).
        userdata: User-specific data passed to the callback.
        msg: The message received from the broker.

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
        microsecond=0,
    )
    logger.info("Received message at %s: %s", t, item)


def query_file(config: FileClient, **kwargs: HumanRSA) -> None:
    """Read a JSON output file and log the entry closest to now.

    Args:
        config: File client configuration with an ``output`` key pointing
            to the JSON file to read.
        **kwargs: Optional keyword arguments. Pass ``receiver`` (RSA key)
            to decrypt entries before processing.

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


def query_mqtt(config: MQTTClient) -> mqtt.Client:
    """Create an MQTT client instance and connect to the configured broker.

    Args:
        config: MQTT client configuration with ``host`` and optional port keys.

    Returns:
        mqtt.Client: Connected MQTT client instance.

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
    if istypedinstance(cast("FileClient", client), FileClient):
        query_file(cast("FileClient", client), receiver=receiver)
    elif istypedinstance(cast("MQTTClient", client), MQTTClient):
        client = query_mqtt(cast("MQTTClient", client))
        client.loop_forever()
