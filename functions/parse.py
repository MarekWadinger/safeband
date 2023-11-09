from argparse import ArgumentParser, FileType, Namespace
from configparser import ConfigParser
from os import getenv
from typing import Union

from pandas import Timedelta
from typing_extensions import NotRequired, TypedDict

from functions.typing_extras import (
    EmailConfig,
    FileClient,
    IOConfig,
    KafkaClient,
    ModelConfig,
    MQTTClient,
    PulsarClient,
    SetupConfig,
)


class Config(TypedDict):
    setup: SetupConfig
    email: NotRequired[EmailConfig]
    model: ModelConfig
    io: IOConfig
    file: NotRequired[FileClient]
    mqtt: NotRequired[MQTTClient]
    kafka: NotRequired[KafkaClient]
    pulsar: NotRequired[PulsarClient]
    client: Union[FileClient, MQTTClient, KafkaClient, PulsarClient, None]


def get_args() -> Namespace:
    """Parses command line arguments.

    Returns:
        Namespace: An object containing the parsed arguments.

    Example:
    >>> import sys
    >>> # Simulate command line arguments
    >>> sys.argv = ['program.py', '-f', 'config.ini', '-r', 'recovery_path',
    ...             '-k', 'key_path', '--threshold', '0.5', '--t-e', '2h',
    ...             '--t-a', '1d', '--t-g', '3w', '-t', 'topic1', 'topic2',
    ...             '--out-topics', 'output1', 'output2', '--path', '/data',
    ...             '--output', '/outs', '--host', 'localhost',
    ...             '--port', '12345', '--bootstrap-servers', 'kafka-server',
    ...             '--service-url', 'pulsar-service', '--debug', 'True']
    >>> args = get_args()
    >>> args.config_file.name
    'config.ini'
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
        "setup", "setup related parameters"
    )
    setup_arg_grp.add_argument(
        "-f", "--config-file", type=FileType("r"), default="config.ini"
    )
    setup_arg_grp.add_argument(
        "-r", "--recovery-path", help="Path to store recovery models"
    )
    setup_arg_grp.add_argument("-k", "--key-path", help="Path to RSA keys")
    setup_arg_grp.add_argument(
        "-d",
        "--debug",
        help="Debug the file using loop as source",
        default=False,
        type=bool,
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
        "--receiver-email",
        type=str,
        help="Receiver email address",
        default=getenv("RECEIVER_EMAIL"),
    )

    model_arg_grp = parser.add_argument_group(
        "model", "Model related parameters"
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
        "file client", "File source related parameters"
    )
    file_arg_grp.add_argument("--path", type=str)
    file_arg_grp.add_argument("--output", type=str)

    mqtt_arg_grp = parser.add_argument_group(
        "mqtt client", "MQTT source related parameters"
    )
    mqtt_arg_grp.add_argument("--host", type=str)
    mqtt_arg_grp.add_argument("--port", type=int)

    kafka_arg_grp = parser.add_argument_group(
        "kafka client", "Kafka source related parameters"
    )
    kafka_arg_grp.add_argument("--bootstrap-servers", type=str)

    pulsar_arg_grp = parser.add_argument_group(
        "pulsar client", "Pulsar source related parameters"
    )
    pulsar_arg_grp.add_argument("--service-url", type=str)

    args = parser.parse_args()

    return args


def get_valid_type(type_) -> type:
    """
    Return a valid type from a given type hint.

    This function takes a type hint and returns a valid Python type that
    can be used to annotate variables and function return types.

    Args:
        type_ (type or typing hint): The type hint to be converted to a valid
        type.

    Returns:
        type: A valid Python type that can be used for type annotations.

    Raises:
        ValueError: Provided type hint is not valid or cannot be converted.

    Example:
        >>> get_valid_type(int)
        <class 'int'>
        >>> get_valid_type(float)
        <class 'float'>
        >>> get_valid_type(str)
        <class 'str'>
        >>> get_valid_type(bool)
        <class 'bool'>
        >>> get_valid_type(list[int])
        list[int]
        >>> get_valid_type(Union[int, float])
        <class 'int'>
        >>> from typing import Optional
        >>> get_valid_type(Optional[str])
        <class 'str'>
        >>> get_valid_type(Union[Timedelta, None])
        <class 'pandas._libs.tslibs.timedeltas.Timedelta'>
        >>> get_valid_type(NotRequired[dict[str, int]])
        <class 'str'>
        >>> get_valid_type(tuple[int, str])
        tuple[int, str]
        >>> get_valid_type(list[Union[int, str]])
        list[typing.Union[int, str]]
        >>> get_valid_type(list[NotRequired[Union[int, str]]])
        list[typing....NotRequired[typing.Union[int, str]]]
        >>> get_valid_type(None)
        Traceback (most recent call last):
        ...
        ValueError: Invalid type: None
    """
    # TODO: get first valid type
    from types import GenericAlias

    if isinstance(type_, (type, GenericAlias)):
        return type_
    elif hasattr(type_, "__args__"):
        # if any([t is type(None) for t in type_.__args__]):
        #     return type(None)
        if "NotRequired" in str(type_):
            return str
        else:
            return get_valid_type(type_.__args__[0])
    else:
        raise ValueError(f"Invalid type: {type_}")


def get_valid_client(config: Config) -> Config:
    """
    Check the validity of the specified client configuration in the given
    'config'.

    The 'config' dictionary contains configuration information for different
    client types such as 'file', 'mqtt', 'kafka', and 'pulsar'. This function
    checks if one and only one client type is specified, and moves the client
    configuration into the 'client' key in 'config'.

    Args:
        config (Dict): A dictionary containing configuration information fo
        different client types.

    Raises:
        ValueError: If multiple or no clients are specified or if the
        configuration is invalid.

    Example:
    >>> config = {
    ...     "setup": {"recovery_path": "/recovery", "key_path": "/keys"},
    ...     "model": {"threshold": 0.5, "t_e": "2h", "t_a": None, "t_g": "1h"},
    ...     "io": {"in_topics": ["t1", "t2"], "out_topics": ["o1"]},
    ...     "mqtt": {"host": "mqtt-server", "port": 1883},
    ... }
    >>> get_valid_client(config)
    {'setup': {'recovery_path': '/recovery', 'key_path': '/keys'},
        'model': {'threshold': 0.5, 't_e': '2h', 't_a': None, 't_g': '1h'},
        'io': {'in_topics': ['t1', 't2'], 'out_topics': ['o1']},
        'client': {'host': 'mqtt-server', 'port': 1883}}

    Multiple clients specified:
    >>> config = {
    ...     "setup": {"recovery_path": "/recovery", "key_path": "/keys"},
    ...     "model": {"threshold": 0.5, "t_e": "2h", "t_a": None, "t_g": "1h"},
    ...     "io": {"in_topics": ["t1", "t2"], "out_topics": ["o1"]},
    ...     "mqtt": {"host": "mqtt-server", "port": 1883},
    ...     "kafka": {"bootstrap_servers": "kafka-server"},
    ... }
    >>> get_valid_client(config)
    Traceback (most recent call last):
    ...
    ValueError: Multiple clients specified: ['mqtt', 'kafka']

    No valid client specified:
    >>> config = {
    ...     "setup": {"recovery_path": "/recovery", "key_path": "/keys"},
    ...     "model": {"threshold": 0.5, "t_e": "2h", "t_a": None, "t_g": "1h"},
    ...     "io": {"in_topics": ["t1", "t2"], "out_topics": ["o1"]},
    ...     "mqtt": {"host": "mqtt-server", "port": None},
    ... }
    >>> get_valid_client(config)
    Traceback (most recent call last):
    ...
    ValueError: Specify one of the clients: ['file', 'mqtt', 'kafka', 'pulsar']
    """
    config_ = config.copy()
    active_clients = []
    clients = ["file", "mqtt", "kafka", "pulsar"]
    for client in clients:
        if client in config_:
            missing_args = any(
                [arg is None for arg in config_[client].values()]
            )
            if missing_args:
                del config_[client]
            else:
                active_clients.append(client)
    if len(active_clients) > 1:
        raise ValueError(f"Multiple clients specified: {active_clients}")
    if len(active_clients) == 0:
        raise ValueError(f"Specify one of the clients: {clients}")
    if len(active_clients) == 1:
        config_["client"] = config_.pop(active_clients[0])

    return config_


def build_config(args: Namespace, config_parser: ConfigParser) -> Config:
    """Builds a configuration dictionary based on command line arguments and a
    configuration file.

    This function constructs a configuration dictionary following the
    structure defined by TypedDicts. It populates the configuration from
    command line arguments when provided, and falls back to values from a
    configuration file.

    Args:
        args (Namespace): Parsed command line arguments.
        config_parser (ConfigParser): The configuration parser for the
        configuration file.

    Returns:
        Config: The configuration dictionary representing the application's
        settings.

    Example:
    >>> from argparse import Namespace
    >>> from configparser import ConfigParser
    >>> args = Namespace(
    ...     recovery_path='/recovery',
    ...     threshold=0.75,
    ...     in_topics=['topic1', 'topic2'],
    ...     path='/data/file.txt'
    ... )
    >>> config_parser = ConfigParser()
    >>> config_parser['setup'] = {'key_path': '/keys', 'debug': 'True'}
    >>> config = build_config(args, config_parser)
    >>> config['setup']['recovery_path']
    '/recovery'
    >>> config['setup']['key_path']
    '/keys'
    >>> config['model']['threshold']
    0.75
    >>> config['io']['in_topics']
    ['topic1', 'topic2']
    >>> config['file']['path']
    '/data/file.txt'
    """
    config_struct = {
        "setup": SetupConfig,
        "email": EmailConfig,
        "model": ModelConfig,
        "io": IOConfig,
        "file": FileClient,
        "mqtt": MQTTClient,
        "kafka": KafkaClient,
        "pulsar": PulsarClient,
    }
    args_ = vars(args)
    config: Config = {}  # type: ignore
    for section, struct in config_struct.items():
        config[section] = {}
        for param, type_ in struct.__annotations__.items():
            type_ = get_valid_type(type_)
            if args_.get(param) is not None:
                param_value = args_[param]
            elif config_parser.has_option(section, param):
                param_value = config_parser[section][param]
            else:
                param_value = None

            if (
                param_value is not None
                and param_value != "None"
                and param_value != ""
            ):
                config[section][param] = type_(param_value)
            elif (
                param_value is None
                or param_value == "None"
                or param_value == ""
            ):
                config[section][param] = None

    return config


def get_params() -> Config:  # pragma: no cover
    """
    Parses command line arguments and a configuration file to create a Config
    object.

    This function combines command line arguments and settings from a
    configuration file to create a configuration object.

    Returns:
        Config: A Config object containing the parsed parameters.
    """

    args = get_args()

    config_parser = ConfigParser()
    config_parser.read_file(args.config_file)

    config = build_config(args, config_parser)

    config = get_valid_client(config)

    return config
