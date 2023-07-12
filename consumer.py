import json
import os

from argparse import Namespace
from datetime import datetime

import paho.mqtt.client as mqtt

from human_security import HumanRSA

from functions.encryption import (
    load_public_key, load_private_key, verify_and_decrypt_data)
from functions.parse import get_argparser, get_config

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
        >>> usr = Namespace(topic="my_topic")
        >>> on_connect(mqtt.Client(), usr, None, 0)
        Connected with result code 0
    """
    print("Connected with result code " + str(rc))
    self.subscribe(userdata.topic)


def on_message(self, userdata, msg):
    """
    MQTT callback function for handling incoming messages.

    Args:
        userdata: User-specific data passed to the callback.
        msg: The message received from the broker.

    Examples:
        >>> obj = mqtt.Client()
        >>> usr = Namespace(topic="my_topic")
        >>> msg = mqtt.MQTTMessage(); msg.payload = b'Hello'
        >>> on_message(obj, usr, msg)
        Received message: Hello
    """
    if isinstance(userdata, Namespace) and 'receiver' in userdata:
        item = verify_and_decrypt_data(json.loads(msg.payload.decode()),
                                       userdata.receiver)
        item = json.dumps(item)
    else:
        item = msg.payload.decode()
    print("Received message: " + item)


def query_file(config: dict, args: Namespace):
    """
    Query a JSON file based on the command-line arguments and print the
    closest past item.

    Args:
        config (dict): The configuration dictionary.
        args (Namespace): Parsed command-line arguments.

    Examples:
        >>> config = {"output": "tests/sample.json"}
        >>> args = Namespace()
        >>> args.date = "2023-01-01 00:00:00"
        >>> query_file(config, args)  # doctest: +ELLIPSIS
        {'time': datetime.datetime(2023, 1, 1, 0, 0), 'anomaly': 0, ...}
    """
    # Load the JSON file as a list of dictionaries
    with open(config.get("output"), 'r') as f:
        data = [json.loads(line) for line in f]

    # Convert the time strings to datetime objects
    for item in data:
        if 'receiver' in args and not item['time'].isascii():
            item = verify_and_decrypt_data(item, args.receiver)
        item["time"] = datetime.strptime(item["time"], "%Y-%m-%d %H:%M:%S")

    # Sort the data by time in descending order
    data.sort(key=lambda x: x["time"], reverse=True)

    # Find the closest past item
    for item in data:
        if item["time"] <= datetime.strptime(args.date, "%Y-%m-%d %H:%M:%S"):
            closest_item = item
            break

    # Print the closest past item
    print(closest_item)


def query_mqtt(config: dict, args: Namespace):
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
        >>> client = query_mqtt(config, args)
        >>> isinstance(client, mqtt.Client)
        True
    """
    # Create MQTT client instance
    client = mqtt.Client(userdata=args)

    # Assign callback functions
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect to the MQTT broker
    client.connect(config.get("host"), PORT, 60)
    return client


if __name__ == '__main__':
    import doctest
    # Run the doctests
    doctest.testmod()

    parser = get_argparser()
    parser.add_argument("-d", "--date", help="Date as 'Y-m-d H:M:S'",
                        default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # Parse command-line arguments
    args = parser.parse_args()

    config = get_config(args.config_file)

    if not os.path.exists(args.key_path):
        raise RuntimeError('Cannot find key path.')
    else:
        args.receiver = HumanRSA()
        load_private_key(args.key_path + "/receiver_pem", args.receiver)
        load_public_key(args.key_path + "/sender_pem.pub", args.receiver)

    if config.get("output"):
        query_file(config, args)
    elif config.get("host"):
        client = query_mqtt(config, args)
        client.loop_forever()
