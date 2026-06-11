import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import pulsar
from streamz import Stream

sys.path.insert(1, str(Path(__file__).parent.parent))
from functions.encryption import (
    decode_data,
    encrypt_data,
    init_rsa_security,
)
from functions.streamz_tools import MapStream  # noqa: F401

logger = logging.getLogger(__name__)


def encryption_service(
    in_topic: list,
    out_topic: str,
    subscription_name: str,
    service_url: str,
) -> None:
    sender, _ = init_rsa_security(".security")

    source = Stream.from_pulsar(
        service_url,
        in_topic,
        subscription_name=subscription_name,
    )

    encrypter = source.map(decode_data).map(encrypt_data, sender)

    if args.out_topic is not None:
        producer = encrypter.to_pulsar(service_url, out_topic)
        L = None
    else:
        L = encrypter.sink_to_list()

    encrypter.start()
    while True:
        try:
            if source.stopped:
                logger.info("Stopping encryption...")
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
        default="my-topic",
        help="The topic to consume messages from. Allows multiply defined.",
        nargs="*",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--out-topic",
        default="dynamic_limits",
        help="The topic to produce messages to.",
        type=str,
    )
    parser.add_argument(
        "--subscription-name",
        default="encryption_service",
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

    encryption_service(
        args.in_topic,
        args.out_topic,
        args.subscription_name,
        args.service_url,
    )
