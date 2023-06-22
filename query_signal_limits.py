import json
import argparse
from datetime import datetime
import paho.mqtt.client as mqtt

from human_security import HumanRSA
from functions.encryption import (
    load_public_key, load_private_key, verify_and_decrypt_data)

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
        >>> usr = argparse.Namespace(topic="my_topic")
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
        >>> usr = argparse.Namespace(topic="my_topic")
        >>> msg = mqtt.MQTTMessage(); msg.payload = b'Hello'
        >>> on_message(obj, usr, msg)
        Received message: Hello
    """
    if isinstance(userdata, argparse.Namespace) and 'reader' in userdata:
        item = verify_and_decrypt_data(json.loads(msg.payload.decode()),
                                       userdata.reader)
        item = json.dumps(item)
    else:
        item = msg.payload.decode()
    print("Received message: " + item)


def parse_args():  # pragma: no cover
    """
    Parse command-line arguments and return the parsed namespace object.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--broker", help="Host or output file path.",
                        default="mqtt.eclipseprojects.io")
    parser.add_argument("-t", "--topic",
                        help="Topic of MQTT or Column of pd.DataFrame",
                        default="test")
    parser.add_argument("-d", "--date", help="Date as 'Y-m-d H:M:S'",
                        default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return parser.parse_args()


def query_file(args):
    """
    Query a JSON file based on the command-line arguments and print the
    closest past item.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Examples:
        >>> args = argparse.Namespace()
        >>> args.broker = "tests/sample.json"
        >>> args.date = "2023-01-01 00:00:00"
        >>> query_file(args)  # doctest: +ELLIPSIS
        {'time': datetime.datetime(2023, 1, 1, 0, 0), 'anomaly': 0, ...}
    """
    # Load the JSON file as a list of dictionaries
    with open(args.broker, 'r') as f:
        data = [json.loads(line) for line in f]

    # Convert the time strings to datetime objects
    for item in data:
        if 'reader' in args and not item['time'].isascii():
            item = verify_and_decrypt_data(item, args.reader)
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


def query_mqtt(args):
    """
    Create an MQTT client instance and connect to the MQTT broker.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        mqtt.Client: MQTT client instance.

    Examples:
        >>> args = argparse.Namespace()
        >>> args.broker = "mqtt.eclipseprojects.io"
        >>> client = query_mqtt(args)
        >>> isinstance(client, mqtt.Client)
        True
    """
    # Create MQTT client instance
    client = mqtt.Client(userdata=args)

    # Assign callback functions
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect to the MQTT broker
    client.connect(args.broker, PORT, 60)
    return client


if __name__ == '__main__':  # pragma: no cover
    import doctest
    # Run the doctests
    doctest.testmod()
    # Parse command-line arguments
    args = parse_args()

    args.reader = HumanRSA()
    args.reader.generate()
    load_public_key("functions/.security/a_pem.pub", args.reader)
    load_private_key("functions/.security/c_pem", args.reader)

    if args.broker.endswith(".json"):
        query_file(args)
    else:
        client = query_mqtt(args)
        client.loop_forever()
