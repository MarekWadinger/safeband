"""MQTT and file-based consumer for anomaly detection results."""

import datetime as dt
import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, cast

import paho.mqtt.client as mqtt
from human_security import HumanRSA

from safeband.encryption import (
    init_rsa_security,
    resolve_key_path,
    verify_and_decrypt_data,
)
from safeband.parse import get_params
from safeband.typing_extras import FileClient, MQTTClient

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
    receiver = getattr(userdata, "receiver", None)
    if receiver is not None:
        decoded = verify_and_decrypt_data(
            json.loads(msg.payload.decode()),
            receiver,
        )
        item = json.dumps(decoded)
        field_count = len(decoded)
    else:
        item = msg.payload.decode()
        field_count = None
    t = dt.datetime.fromtimestamp(msg.timestamp, tz=dt.UTC).replace(
        microsecond=0,
    )
    # Log only metadata at INFO; the full decrypted payload may carry
    # sensitive values, so it is emitted at DEBUG instead.
    logger.info(
        "Received message at %s on %s (%s fields)",
        t,
        msg.topic,
        field_count if field_count is not None else "n/a",
    )
    logger.debug("Message payload at %s: %s", t, item)


def query_file(config: FileClient, **kwargs: HumanRSA | None) -> None:
    """Read a JSON output file and log the entry closest to now.

    Args:
        config: File client configuration with an ``output`` key pointing
            to the JSON file to read.
        **kwargs: Optional keyword arguments. Pass ``receiver`` (RSA key)
            to decrypt entries before processing; with no key (or
            ``receiver=None``) entries are treated as plaintext.

    """
    receiver = kwargs.get("receiver")
    # Load the JSON file as a list of dictionaries
    with Path(config.output).open(encoding="utf-8") as f:
        data: list[dict[str, Any]] = [json.loads(line) for line in f]

    # Convert the time strings to datetime objects
    for i, item in enumerate(data):
        # Encrypted entries are detected by their signature field rather
        # than by guessing from the ciphertext's character set.
        if receiver is not None and "signature" in item:
            data[i] = cast(
                "dict[str, Any]",
                verify_and_decrypt_data(item, receiver),
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

    # Log only metadata at INFO; the full (possibly decrypted) entry may
    # carry sensitive values, so it is emitted at DEBUG instead.
    if closest_item is not None:
        logger.info(
            "Closest entry at %s (%s fields)",
            closest_item["time"],
            len(closest_item),
        )
    else:
        logger.info("No entry at or before now.")
    logger.debug("Closest entry payload: %s", closest_item)


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
    client.connect(config.host, PORT, 60)
    return client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = get_params()

    receiver: HumanRSA | None = None
    if config.setup.key_path:
        safe_key_path = resolve_key_path(config.setup.key_path)
        _, receiver = init_rsa_security(str(safe_key_path))

    client = config.client
    if isinstance(client, FileClient):
        query_file(client, receiver=receiver)
    elif isinstance(client, MQTTClient):
        mqtt_client = query_mqtt(client)
        mqtt_client.loop_forever()
