import sys
from argparse import ArgumentParser
from pathlib import Path

import pulsar
from pulsar.schema import JsonSchema, Record, String
from streamz import Stream

sys.path.insert(1, str(Path(__file__).parent.parent))
from functions.encryption import (  # noqa: E402
    decrypt_data,
    encode_data,
    init_rsa_security,
)
from functions.streamz_tools import map  # noqa: E402, F401


class Example(Record):
    # keys and __getitem__ serve as minimum implementation of mapping protocol
    def keys(self):
        return self._fields.keys()

    def __getitem__(self, key):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["_required", "_default", "_required_default"]
        }[key]

    time = String()
    anomaly = String()
    level_high = String()
    level_low = String()


def decryption_service(
    in_topic: list, out_topic: str, subscription_name: str, service_url: str
):
    _, receiver = init_rsa_security(".security")

    source = Stream.from_pulsar(
        service_url,
        in_topic,
        subscription_name=subscription_name,
        consumer_params={"schema": JsonSchema(Example)},
    )
    source.map(lambda x: x.decode())
    decrypter = source.map(dict).map(encode_data).map(decrypt_data, receiver)

    if args.out_topic is not None:
        producer = decrypter.to_pulsar(
            service_url,
            out_topic,
        )
        L = None
    else:
        L = decrypter.sink_to_list()

    decrypter.start()
    while True:
        try:
            if source.stopped:
                print("Stopping decryption...")
                break
            if L:
                print(L.pop(0))
        except pulsar.Interrupted:
            print("Stop receiving messages")
            if args.out_topic is not None:
                producer.stop()
                producer.flush()
            break
        except Exception as e:
            raise e


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--in-topic",
        default="dynamic_limits",
        help="The topic to consume messages from. Allows multiply defined.",
        nargs="*",
        type=str,
    )
    parser.add_argument(
        "-o", "--out-topic", help="The topic to produce messages to.", type=str
    )
    parser.add_argument(
        "--subscription-name",
        default="decryption_service",
        help="Name consumer's subscription.",
        type=str,
    )
    parser.add_argument(
        "--service-url",
        default="pulsar://localhost:6650",
        help="The scheme and broker as 'scheme://IP:port.",
        type=str,
    )
    args = parser.parse_args()

    decryption_service(
        args.in_topic, args.out_topic, args.subscription_name, args.service_url
    )
