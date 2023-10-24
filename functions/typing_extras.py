from typing import Union

from pandas import Timedelta
from typing_extensions import NotRequired, TypedDict


class FileClient(TypedDict):
    path: str
    output: str


class MQTTClient(TypedDict):
    host: str
    port: int


class KafkaClient(TypedDict):
    bootstrap_servers: str


class PulsarClient(TypedDict):
    service_url: str


class IOConfig(TypedDict):
    in_topics: list[str]
    out_topics: Union[list[str], str, None]


class ModelConfig(TypedDict):
    threshold: float
    t_e: Timedelta
    t_a: Union[Timedelta, None]
    t_g: Union[Timedelta, None]


class SetupConfig(TypedDict):
    recovery_path: NotRequired[str]
    key_path: NotRequired[str]
    debug: NotRequired[bool]


def istypedinstance(obj, type_):
    """
    Checks if the given object matches the provided type annotation.

    This function checks if the object `obj` is an instance that conforms
    to the specified type annotation `type_`.

    Args:
        obj (dict): The object to be checked.
        type_ (type): The type annotation to compare against.

    Returns:
        bool: True if the object matches the provided type annotation; False
        otherwise.

    Example:
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
                "NotRequired" in str(property_type) or
                str(type(property_type)) ==
                "<class 'typing._GenericAlias'>"):
            if hasattr(property_type, "__args__"):
                property_type = Union[property_type.__args__[0], None]
            else:
                property_type = type(None)
        try:
            return isinstance(value, property_type)
        except TypeError:
            if hasattr(property_type, "__args__"):
                return isinstance(
                    value, (property_type.__args__[0], type(None)))
            else:
                return False
    return True
