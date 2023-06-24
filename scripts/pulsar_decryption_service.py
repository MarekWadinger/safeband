import pulsar
import sys

from argparse import ArgumentParser
from pathlib import Path

from streamz import Stream

sys.path.insert(1, str(Path(__file__).parent.parent))
from functions.encryption import (  # noqa: E402
    init_rsa_security, decrypt_data)


def decryption_service(
        in_topic: list,
        out_topic: str,
        subscription_name: str,
        service_url: str
        ):
    _, receiver = init_rsa_security(".security")

    source = Stream.from_pulsar(
        in_topic,
        subscription_name=subscription_name,
        consumer_params={'service_url': service_url})

    source.map(lambda x: x.decode())
    decrypter = (
        source
        .map(decrypt_data, receiver)
        )

    if args.out_topic is not None:
        producer = decrypter.to_pulsar(
            out_topic,
            producer_config={"service_url": service_url})
    else:
        L = decrypter.sink_to_list()

    decrypter.start()
    while True:
        try:
            if L:
                print(L.pop(0))
        except pulsar.Interrupted:
            print("Stop receiving messages")
            if args.out_topic is not None:
                producer.stop()
                producer.flush()
            break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--in-topic',
        default="my-output",
        help="The topic to consume messages from. Allows multiply defined.",
        nargs='*', type=str)
    parser.add_argument(
        '-o', '--out-topic',
        help="The topic to produce messages to.",
        type=str)
    parser.add_argument(
        '--subscription-name',
        default="decryption_service",
        help="Name consumer's subscription.",
        type=str)
    parser.add_argument(
        '--service-url',
        default="pulsar://localhost:6650",
        help="The scheme and broker as 'scheme://IP:port.",
        type=str)
    args = parser.parse_args()

    decryption_service(args.in_topic, args.out_topic, args.subscription_name,
                       args.service_url)
