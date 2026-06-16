"""Argument parsing and configuration building for the consumer pipeline."""

from __future__ import annotations

from argparse import (
    ArgumentParser,
    FileType,  # ty: ignore[deprecated]
    Namespace,
)
from configparser import ConfigParser
from os import getenv
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from pandas import Timedelta

from functions.typing_extras import (
    Config,
    FileClient,
    KafkaClient,
    MQTTClient,
    NATSClient,
    PulsarClient,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

# Fields of each config section, used to gather values from CLI args and the
# config file. Mirrors the (non-client) fields of the Pydantic models.
_SECTION_FIELDS: dict[str, list[str]] = {
    "setup": ["recovery_path", "key_path", "debug"],
    "email": ["sender_email", "sender_password", "recipient_email"],
    "model": ["threshold", "t_e", "t_a", "t_g", "physical_limits"],
    "io": ["in_topics", "out_topics"],
    "file": ["path", "output"],
    "mqtt": ["host", "port"],
    "kafka": ["bootstrap_servers"],
    "pulsar": ["service_url"],
    "nats": ["servers"],
}

# Transport sections, in dispatch precedence order.
_CLIENTS: list[str] = ["file", "mqtt", "kafka", "pulsar", "nats"]

_CLIENT_MODELS: dict[str, type[BaseModel]] = {
    "file": FileClient,
    "mqtt": MQTTClient,
    "kafka": KafkaClient,
    "pulsar": PulsarClient,
    "nats": NATSClient,
}


def get_args() -> Namespace:
    """Parse command line arguments.

    Returns:
        Namespace: An object containing the parsed arguments.

    Example:
    >>> import sys
    >>> # Simulate command line arguments
    >>> sys.argv = ['program.py', '-f', 'example.ini', '-r', 'recovery_path',
    ...             '-k', 'key_path', '--threshold', '0.5', '--t-e', '2h',
    ...             '--t-a', '1d', '--t-g', '3w', '-t', 'topic1', 'topic2',
    ...             '--out-topics', 'output1', 'output2', '--path', '/data',
    ...             '--output', '/outs', '--host', 'localhost',
    ...             '--port', '12345', '--bootstrap-servers', 'kafka-server',
    ...             '--service-url', 'pulsar-service', '--debug']
    >>> args = get_args()
    >>> args.config_file.name
    'example.ini'
    >>> args.recovery_path
    'recovery_path'
    >>> args.key_path
    'key_path'
    >>> args.debug
    True
    >>> args.threshold
    0.5
    >>> args.t_e
    Timedelta('0 days 02:00:00')
    >>> args.t_a
    Timedelta('1 days 00:00:00')
    >>> args.t_g
    Timedelta('21 days 00:00:00')
    >>> args.in_topics
    ['topic1', 'topic2']
    >>> args.out_topics
    ['output1', 'output2']
    >>> args.path
    '/data'
    >>> args.output
    '/outs'
    >>> args.host
    'localhost'
    >>> args.port
    12345
    >>> args.bootstrap_servers
    'kafka-server'
    >>> args.service_url
    'pulsar-service'

    """
    parser = ArgumentParser()

    setup_arg_grp = parser.add_argument_group(
        "setup",
        "setup related parameters",
    )

    def file_or_none(value: str | None) -> IO[str] | None:
        if value is not None and Path(value).is_file():
            return FileType("r")(value)  # ty: ignore[deprecated]
        return None

    setup_arg_grp.add_argument(
        "-f",
        "--config-file",
        type=file_or_none,
        default="config.ini",
    )
    setup_arg_grp.add_argument(
        "-r",
        "--recovery-path",
        help="Path to store recovery models",
    )
    setup_arg_grp.add_argument("-k", "--key-path", help="Path to RSA keys")
    setup_arg_grp.add_argument(
        "-d",
        "--debug",
        help="Debug the file using loop as source",
        action="store_true",
        default=None,
    )

    mail_arg_grp = parser.add_argument_group("mail")
    mail_arg_grp.add_argument(
        "--sender-email",
        type=str,
        help="Senders email address",
        default=getenv("SENDER_EMAIL"),
    )
    mail_arg_grp.add_argument(
        "--sender-password",
        type=str,
        help="Senders password",
        default=getenv("SENDER_PASSWORD"),
    )
    mail_arg_grp.add_argument(
        "--recipient-email",
        type=str,
        help="Recipient email address",
        default=getenv("RECIPIENT_EMAIL"),
    )

    model_arg_grp = parser.add_argument_group(
        "model",
        "Model related parameters",
    )
    model_arg_grp.add_argument("--threshold", type=float)
    model_arg_grp.add_argument("--t-e", type=Timedelta)
    model_arg_grp.add_argument("--t-a", type=Timedelta)
    model_arg_grp.add_argument("--t-g", type=Timedelta)

    io_arg_grp = parser.add_argument_group("io")
    io_arg_grp.add_argument(
        "-t",
        "--in-topics",
        nargs="*",
        type=str,
        help="Topic of MQTT or Column of pd.DataFrame",
    )
    io_arg_grp.add_argument("--out-topics", nargs="*", type=str)

    file_arg_grp = parser.add_argument_group(
        "file client",
        "File source related parameters",
    )
    file_arg_grp.add_argument("--path", type=str)
    file_arg_grp.add_argument("--output", type=str)

    mqtt_arg_grp = parser.add_argument_group(
        "mqtt client",
        "MQTT source related parameters",
    )
    mqtt_arg_grp.add_argument("--host", type=str)
    mqtt_arg_grp.add_argument("--port", type=int)

    kafka_arg_grp = parser.add_argument_group(
        "kafka client",
        "Kafka source related parameters",
    )
    kafka_arg_grp.add_argument("--bootstrap-servers", type=str)

    pulsar_arg_grp = parser.add_argument_group(
        "pulsar client",
        "Pulsar source related parameters",
    )
    pulsar_arg_grp.add_argument("--service-url", type=str)

    nats_arg_grp = parser.add_argument_group(
        "nats client",
        "NATS source related parameters",
    )
    nats_arg_grp.add_argument("--servers", type=str)

    return parser.parse_args()


def _gather_section(
    section: str,
    fields: list[str],
    args_: dict[str, Any],
    config_parser: ConfigParser,
) -> dict[str, Any]:
    """Merge CLI args and config-file options for a single config section.

    CLI values win; an arg that is ``None`` falls back to the config-file
    option, then to ``None``. The literal strings ``"None"`` and ``""``
    coming from a config file are normalised to ``None``.

    Args:
        section: Section name (for example ``"model"``).
        fields: Field names belonging to the section.
        args_: Parsed CLI arguments as a mapping.
        config_parser: The configuration file parser.

    Returns:
        dict: Mapping of field name to its resolved value (or ``None``).

    """
    gathered: dict[str, Any] = {}
    for field in fields:
        if args_.get(field) is not None:
            value: Any = args_[field]
        elif config_parser.has_option(section, field):
            value = config_parser[section][field]
        else:
            value = None
        if value in ("None", ""):
            value = None
        gathered[field] = value
    return gathered


def get_valid_client(config: Config) -> Config:
    """Resolve the single active transport client in ``config``.

    Exactly one fully specified transport section (``file``, ``mqtt``,
    ``kafka``, ``pulsar`` or ``nats``) must be present; that section is
    moved into ``config.client``. A section whose required fields are not
    all specified is treated as absent.

    Args:
        config: The configuration whose client sections are inspected.

    Returns:
        Config: The same configuration with ``client`` set to the resolved
        transport model.

    Raises:
        ValueError: If multiple clients are fully specified or none is.

    Example:
    >>> config = build_config(
    ...     Namespace(host="mqtt-server", port=1883),
    ...     ConfigParser(),
    ... )
    >>> config = get_valid_client(config)
    >>> config.client
    MQTTClient(host='mqtt-server', port=1883)

    Multiple clients specified:
    >>> config = build_config(
    ...     Namespace(
    ...         host="mqtt-server", port=1883,
    ...         bootstrap_servers="kafka-server",
    ...     ),
    ...     ConfigParser(),
    ... )
    >>> get_valid_client(config)
    Traceback (most recent call last):
    ...
    ValueError: Multiple clients specified: ['mqtt', 'kafka']

    No valid client specified:
    >>> config = build_config(Namespace(host="mqtt-server"), ConfigParser())
    >>> get_valid_client(config)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Specify one of the clients: [...]

    """
    active_clients = [
        client for client in _CLIENTS if getattr(config, client) is not None
    ]
    if len(active_clients) > 1:
        msg = f"Multiple clients specified: {active_clients}"
        raise ValueError(msg)
    if len(active_clients) == 0:
        msg = f"Specify one of the clients: {_CLIENTS}"
        raise ValueError(msg)
    config.client = getattr(config, active_clients[0])
    return config


def build_config(args: Namespace, config_parser: ConfigParser) -> Config:
    """Build and validate a :class:`Config` from CLI args and a config file.

    Values are merged section by section, preferring CLI arguments and
    falling back to the config file. The resulting nested mapping is then
    validated into a Pydantic :class:`Config`; a transport client section
    is only constructed when all of its required fields are present.

    Args:
        args (Namespace): Parsed command line arguments.
        config_parser (ConfigParser): The configuration parser for the
        configuration file.

    Returns:
        Config: The validated configuration model.

    Example:
    >>> from argparse import Namespace
    >>> from configparser import ConfigParser
    >>> args = Namespace(
    ...     recovery_path='/recovery',
    ...     threshold=0.75,
    ...     in_topics=['topic1', 'topic2'],
    ...     path='/data/file.txt',
    ...     output='/data/out.json',
    ... )
    >>> config_parser = ConfigParser()
    >>> config_parser['setup'] = {'key_path': '/keys', 'debug': 'True'}
    >>> config = build_config(args, config_parser)
    >>> config.setup.recovery_path
    '/recovery'
    >>> config.setup.key_path
    '/keys'
    >>> config.setup.debug
    True
    >>> config.model.threshold
    0.75
    >>> config.io.in_topics
    ['topic1', 'topic2']
    >>> config.file.path
    '/data/file.txt'

    """
    args_ = vars(args)
    raw: dict[str, Any] = {}
    for section, fields in _SECTION_FIELDS.items():
        gathered = _gather_section(section, fields, args_, config_parser)
        if section in _CLIENT_MODELS:
            # A transport section is only kept when every required field
            # is specified; partial sections are treated as absent so the
            # exactly-one-client rule can be enforced downstream.
            model = _CLIENT_MODELS[section]
            required = [
                name
                for name, info in model.model_fields.items()
                if info.is_required()
            ]
            if any(gathered.get(name) is None for name in required):
                continue
        # Drop absent fields (gathered as None) so the model's own field
        # defaults apply instead of forcing None onto required fields.
        raw[section] = {k: v for k, v in gathered.items() if v is not None}
    return Config.model_validate(raw)


def get_params() -> Config:  # pragma: no cover
    """Parse command line arguments and config file into a Config object.

    This function combines command line arguments and settings from a
    configuration file to create a configuration object.

    Returns:
        Config: A Config object containing the parsed parameters.

    """
    args = get_args()

    config_parser = ConfigParser()
    if args.config_file:
        config_parser.read_file(args.config_file)

    config = build_config(args, config_parser)

    return get_valid_client(config)
