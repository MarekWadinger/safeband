"""Pydantic v2 models describing the consumer configuration sections.

Each section of the application config (``[setup]``, ``[model]``, ``[io]``
and the per-transport client sections) is modelled as a Pydantic v2
``BaseModel``. Validation happens by construction, replacing the previous
hand-rolled ``istypedinstance`` checker (which only validated the first
field of a TypedDict). Transport dispatch is done with ``isinstance`` on
the distinct client models.
"""

from __future__ import annotations

import json

from pandas import Timedelta
from pydantic import BaseModel, ConfigDict, field_validator


class EmailConfig(BaseModel):
    """SMTP credentials and recipient for outgoing alert emails."""

    sender_email: str | None = None
    sender_password: str | None = None
    recipient_email: str | None = None


class FileClient(BaseModel):
    """File-based client configuration with input path and output path."""

    path: str
    output: str


class MQTTClient(BaseModel):
    """MQTT broker connection parameters."""

    model_config = ConfigDict(extra="allow")

    host: str
    port: int


class KafkaClient(BaseModel):
    """Kafka consumer/producer connection parameters."""

    model_config = ConfigDict(extra="allow")

    bootstrap_servers: str


class PulsarClient(BaseModel):
    """Apache Pulsar client connection parameters."""

    model_config = ConfigDict(extra="allow")

    service_url: str


class NATSClient(BaseModel):
    """NATS client connection parameters.

    The ``servers`` field holds one or more NATS URLs (for example
    ``nats://localhost:4222``). Multiple servers may be supplied as a
    single comma-separated string; the source and sink split it into a
    ``list[str]`` before handing it to ``nats.connect``.
    """

    model_config = ConfigDict(extra="allow")

    servers: str


class IOConfig(BaseModel):
    """Input and output topic names for the messaging pipeline."""

    in_topics: list[str] = []
    out_topics: list[str] | str | None = None


class ModelConfig(BaseModel):
    """Anomaly model hyper-parameters: thresholds and time windows.

    ``t_e``/``t_a``/``t_g`` accept a :class:`pandas.Timedelta` or a string
    such as ``"2h"``/``"1d"`` and are coerced to ``Timedelta``.
    ``physical_limits`` accepts a JSON object string (as found in config
    files) or an already-built mapping and is parsed to a mapping in code.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    threshold: float | None = None
    t_e: Timedelta | None = None
    t_a: Timedelta | None = None
    t_g: Timedelta | None = None
    # JSON object string in config files, parsed value in code. The strict
    # mapping/(low, high) shape is enforced where the value is consumed
    # (rpc_server._parse_physical_limits), so the loaded JSON is accepted
    # loosely here and a malformed shape fails at use, not construction.
    physical_limits: object | None = None

    @field_validator("t_e", "t_a", "t_g", mode="before")
    @classmethod
    def _coerce_timedelta(cls, value: object) -> Timedelta | None:
        """Coerce a string such as ``"2h"`` into a pandas ``Timedelta``."""
        if value is None or isinstance(value, Timedelta):
            return value
        # Pydantic feeds the raw config value (typically a "2h"-style
        # string); Timedelta accepts str/number and raises on anything
        # else, which Pydantic surfaces as a validation error.
        return Timedelta(value)  # ty: ignore[invalid-argument-type]

    @field_validator("physical_limits", mode="before")
    @classmethod
    def _parse_physical_limits(cls, value: object) -> object:
        """Parse a JSON object string into a mapping; pass dict/None on."""
        if isinstance(value, str):
            return json.loads(value)
        return value


class SetupConfig(BaseModel):
    """Optional infrastructure paths and debug flag for the consumer setup."""

    recovery_path: str | None = None
    key_path: str | None = None
    debug: bool | None = None


class Config(BaseModel):
    """Top-level configuration for the consumer application."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    setup: SetupConfig
    email: EmailConfig | None = None
    model: ModelConfig
    io: IOConfig
    file: FileClient | None = None
    mqtt: MQTTClient | None = None
    kafka: KafkaClient | None = None
    pulsar: PulsarClient | None = None
    nats: NATSClient | None = None
    client: (
        FileClient
        | MQTTClient
        | KafkaClient
        | PulsarClient
        | NATSClient
        | None
    ) = None
