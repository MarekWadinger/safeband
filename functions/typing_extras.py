"""TypedDict definitions and instance-checking utilities for config types."""

from collections.abc import Mapping
from typing import NotRequired

from pandas import Timedelta
from typing_extensions import TypedDict


class EmailConfig(TypedDict):
    """SMTP credentials and recipient for outgoing alert emails."""

    sender_email: str
    sender_password: str
    recipient_email: str


class FileClient(TypedDict):
    """File-based client configuration with input path and output path."""

    path: str
    output: str


class MQTTClient(TypedDict):
    """MQTT broker connection parameters."""

    host: str
    port: int


class KafkaClient(TypedDict):
    """Kafka consumer/producer connection parameters."""

    bootstrap_servers: str


class PulsarClient(TypedDict):
    """Apache Pulsar client connection parameters."""

    service_url: str


class IOConfig(TypedDict):
    """Input and output topic names for the messaging pipeline."""

    in_topics: list[str]
    out_topics: list[str] | str | None


class ModelConfig(TypedDict):
    """Anomaly model hyper-parameters: thresholds and time windows."""

    threshold: float
    t_e: Timedelta
    t_a: Timedelta | None
    t_g: Timedelta | None
    # JSON object string in config files, parsed mapping in code.
    physical_limits: NotRequired[str | dict[str, tuple[float, float]] | None]


class SetupConfig(TypedDict):
    """Optional infrastructure paths and debug flag for the consumer setup."""

    recovery_path: NotRequired[str]
    key_path: NotRequired[str]
    debug: NotRequired[bool]


def istypedinstance(obj: Mapping[str, object], type_: type) -> bool:
    """Check if the given object matches the provided type annotation.

    This function checks if the object `obj` is an instance that conforms
    to the specified type annotation `type_`.

    Args:
        obj (dict): The object to be checked.
        type_ (type): The type annotation to compare against.

    Returns:
        bool: True if the object matches the provided type annotation; False
        otherwise.

    Examples:
    # >>> model_config = {
    # ...     'threshold': 0.5, 't_e': Timedelta('1 days'),
    # ...     't_a': None, 't_g': None}
    # >>> istypedinstance(model_config, ModelConfig)
    # True

    >>> setup_config = {
    ...     'recovery_path': "./key"}
    >>> istypedinstance(setup_config, SetupConfig)
    True

    >>> setup_config = {
    ...     'recovery_path': 5}
    >>> istypedinstance(setup_config, SetupConfig)
    False

    """
    for property_name, property_type in type_.__annotations__.items():
        value = obj.get(property_name, None)
        if (
            "NotRequired" in str(property_type)
            or str(type(property_type)) == "<class 'typing._GenericAlias'>"
        ):
            if hasattr(property_type, "__args__"):
                resolved_type = property_type.__args__[0] | None
            else:
                resolved_type = type(None)
        else:
            resolved_type = property_type
        try:
            return isinstance(value, resolved_type)
        except TypeError:
            if hasattr(property_type, "__args__"):
                return isinstance(
                    value,
                    (property_type.__args__[0], type(None)),
                )
            return False
    return True
