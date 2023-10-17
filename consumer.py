import datetime as dt
import json
from argparse import Namespace
from datetime import datetime

import paho.mqtt.client as mqtt

from functions.encryption import (
    init_rsa_security,
    verify_and_decrypt_data,
)
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
    if isinstance(userdata, Namespace) and 'receiver' in userdata:
        item = verify_and_decrypt_data(json.loads(msg.payload.decode()),
                                       userdata.receiver)
        item = json.dumps(item)
    else:
        item = msg.payload.decode()
    t = dt.datetime.fromtimestamp(msg.timestamp).replace(microsecond=0)
    print(f"Received message at {t}: {item}")


def query_file(config: dict[str, str], args: Namespace):
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
        >>> query_file(config, args)
        {'time': datetime.datetime(2023, 1, 1, 0, 0), 'anomaly': 0, ...}
    """
    # Load the JSON file as a list of dictionaries
    with open(config.get("output", ""), encoding='utf-8') as f:
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


def query_mqtt(config: dict[str, str], args: Namespace):
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
    client.connect(config.get("host", ""), PORT, 60)
    return client


if __name__ == '__main__':
    parser = get_argparser()
    parser.add_argument(
        "-d", "--date", help="Date as 'Y-m-d H:M:S'",
        default=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    # Parse command-line arguments
    args_ = parser.parse_args()

    config_ = get_config(args_.config_file)

    if args_.key_path:
        _, args_.receiver = init_rsa_security(args_.key_path)

    if config_.get("output"):
        query_file(config_, args_)
    elif config_.get("host"):
        client = query_mqtt(config_, args_)
        client.loop_forever()
