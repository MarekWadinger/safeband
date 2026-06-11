"""Pulsar consumer that decrypts RSA-encrypted anomaly detection results."""

import logging
import sys
from argparse import ArgumentParser
from collections.abc import KeysView
from pathlib import Path

import pulsar
from pulsar.schema import JsonSchema, Record, String
from streamz import Stream

sys.path.insert(1, str(Path(__file__).parent.parent))
from functions.encryption import (
    decrypt_data,
    encode_data,
    init_rsa_security,
)
from functions.streamz_tools import MapStream  # noqa: F401

logger = logging.getLogger(__name__)


class Example(Record):
    """Pulsar schema record mirroring the anomaly detection output fields."""

    # keys and __getitem__ serve as minimum implementation of mapping protocol
    def keys(self) -> KeysView[str]:
        """Return the declared schema field names."""
        return self._fields.keys()  # ty: ignore[unresolved-attribute]

    def __getitem__(self, key: str) -> object:
        """Return the value for the given field name."""
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
    in_topic: list,
    out_topic: str,
    subscription_name: str,
    service_url: str,
) -> None:
    """Subscribe to a Pulsar topic, decrypt messages, and forward or print them.

    Args:
        in_topic: Pulsar topics to consume from.
        out_topic: Pulsar topic to publish decrypted messages to, or None to
            print to stdout.
        subscription_name: Consumer subscription name.
        service_url: Pulsar broker URL.

    """
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
                logger.info("Stopping decryption...")
                break
            if L:
                logger.info("%s", L.pop(0))
        except pulsar.Interrupted:
            logger.info("Stop receiving messages")
            if args.out_topic is not None:
                producer.stop()
                producer.flush()
            break
        except Exception:
            raise


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
        "-o",
        "--out-topic",
        help="The topic to produce messages to.",
        type=str,
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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parser.parse_args()

    decryption_service(
        args.in_topic,
        args.out_topic,
        args.subscription_name,
        args.service_url,
    )
