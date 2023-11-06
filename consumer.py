import datetime as dt
import json
from argparse import Namespace

import paho.mqtt.client as mqtt

from functions.encryption import init_rsa_security, verify_and_decrypt_data
from functions.parse import get_params
from functions.typing_extras import FileClient, MQTTClient, istypedinstance

PORT = 1883


# MQTT callback functions
def on_connect(self: mqtt.Client, userdata, flags, rc):
    """
    MQTT callback function for handling the connect event.

    Args:
        userdata: User-specific data passed to the callback.
        flags: Response flags from the broker.
        rc: The connection result code.

    Examples:
        >>> obj = mqtt.Client()
        >>> usr = Namespace(topic=["my_topic"])
        >>> on_connect(mqtt.Client(), usr, None, 0)
        Connected with result code 0
    """
    print("Connected with result code " + str(rc))
    self.subscribe([(topic, 0) for topic in userdata.topic])


def on_message(self, userdata, msg):
    """
    MQTT callback function for handling incoming messages.

    Args:
        userdata: User-specific data passed to the callback.
        msg: The message received from the broker.

    Examples:
        >>> obj = mqtt.Client()
        >>> usr = Namespace(topic=["my_topic"])
        >>> msg = mqtt.MQTTMessage(); msg.payload = b'Hello'
        >>> on_message(obj, usr, msg)
        Received message at 1970-01-01 ...: Hello
    """
    if isinstance(userdata, Namespace) and "receiver" in userdata:
        item = verify_and_decrypt_data(
            json.loads(msg.payload.decode()), userdata.receiver
        )
        item = json.dumps(item)
    else:
        item = msg.payload.decode()
    t = dt.datetime.fromtimestamp(msg.timestamp).replace(microsecond=0)
    print(f"Received message at {t}: {item}")


def query_file(config: FileClient, **kwargs):
    """
    Query a JSON file based on the command-line arguments and print the
    closest past item.

    Args:
        config (dict): The configuration dictionary.
        args (Namespace): Parsed command-line arguments.

    Examples:
        >>> config = {"output": "tests/sample.json"}
        >>> query_file(config)
        {'time': datetime.datetime(2023, 1, 1, 0, 0), 'anomaly': 0, ...}
    """
    # Load the JSON file as a list of dictionaries
    with open(config.get("output", ""), encoding="utf-8") as f:
        data: list[dict] = [json.loads(line) for line in f]

    # Convert the time strings to datetime objects
    for item in data:
        if "receiver" in kwargs and not item["time"].isascii():
            item = verify_and_decrypt_data(item, kwargs["receiver"])
        item["time"] = dt.datetime.strptime(item["time"], "%Y-%m-%d %H:%M:%S")

    # Sort the data by time in descending order
    data.sort(key=lambda x: x["time"], reverse=True)

    # Find the closest past item
    for item in data:
        if item["time"] <= dt.datetime.strptime(
            dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "%Y-%m-%d %H:%M:%S",
        ):
            closest_item = item
            break

    # Print the closest past item
    print(closest_item)


def query_mqtt(config: MQTTClient):
    """
    Create an MQTT client instance and connect to the MQTT broker.

    Args:
        config (dict): The configuration dictionary.
        args (Namespace): Parsed command-line arguments.

    Returns:
        mqtt.Client: MQTT client instance.

    Examples:
        >>> config = {"host": "mqtt.eclipseprojects.io"}
        >>> args = Namespace()
        >>> client = query_mqtt(config)
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
    config = get_params()

    if "key_path" in config["setup"]:
        _, receiver = init_rsa_security(config["setup"]["key_path"])

    if istypedinstance(config["client"], FileClient):
        query_file(config["client"])
    elif istypedinstance(config["client"], MQTTClient):
        client = query_mqtt(config["client"])
        client.loop_forever()
