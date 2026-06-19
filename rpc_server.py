"""Streaming anomaly detection server with MQTT, Kafka, and file I/O."""

# IMPORTS
import contextlib
import datetime as dt
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, cast

import pandas as pd
from paho.mqtt.client import MQTTMessage
from river import proba, utils
from streamz import Stream

try:
    from pulsar.schema import JsonSchema, Record
    from pulsar.schema import String as PulsarString

    _PULSAR_AVAILABLE = True
except ImportError:
    _PULSAR_AVAILABLE = False

from safeband.anomaly import ConditionalGaussianScorer, GaussianScorer
from safeband.email_client import EmailClient
from safeband.encryption import (
    encrypt_and_sign_data,
    init_rsa_security,
)
from safeband.model_persistence import load_model, save_model
from safeband.proba import MultivariateGaussian
from safeband.streamz_tools import (  # noqa: F401
    _filt,
    _func,
    from_nats,
    to_mqtt,
    to_nats,
)
from safeband.typing_extras import (
    EmailConfig,
    FileClient,
    IOConfig,
    KafkaClient,
    ModelConfig,
    MQTTClient,
    NATSClient,
    PulsarClient,
    SetupConfig,
)
from safeband.utils import common_prefix

logger = logging.getLogger(__name__)

# CONSTANTS

_exit_stack: contextlib.ExitStack = contextlib.ExitStack()

# Union of every supported transport configuration accepted by the
# source/sink dispatch methods.
_ClientConfig = (
    FileClient | MQTTClient | KafkaClient | PulsarClient | NATSClient
)


# DEFINITIONS
def _parse_physical_limits(
    value: object,
) -> dict[str, tuple[float, float]] | None:
    """Parse a physical_limits config entry into per-signal bounds.

    Accepts ``None``, a JSON object string, or an already-built mapping
    of signal name to a (low, high) pair.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        msg = (
            "physical_limits must be a mapping of signal name to "
            f"(low, high); got {value!r}"
        )
        raise TypeError(msg)
    limits: dict[str, tuple[float, float]] = {}
    items = cast("dict[Any, Any]", value).items()
    for name, bounds in items:
        try:
            phys_low, phys_high = bounds
        except (TypeError, ValueError) as exc:
            msg = (
                "physical_limits bounds must be a (low, high) pair; "
                f"got {bounds!r} for signal {name!r}"
            )
            raise ValueError(msg) from exc
        limits[str(name)] = (float(phys_low), float(phys_high))
    return limits


def expand_model_params(
    model_params: ModelConfig,
) -> tuple[
    float,
    dt.timedelta,
    dt.timedelta,
    dt.timedelta,
    dict[str, tuple[float, float]] | None,
]:
    """Extract and convert model parameters from the configuration dictionary.

    Args:
        model_params: Mapping containing threshold, t_e, t_a, t_g, and
            optional physical_limits values.

    Returns:
        tuple: ``(threshold, t_e, t_a, t_g, physical_limits)`` as a float,
        three timedeltas, and an optional mapping of signal name to its
        static (low, high) operating bounds.

    Examples:
        >>> *_, limits = expand_model_params(ModelConfig(
        ...     t_e=pd.Timedelta("1d"),
        ...     physical_limits='{"plant/a": [0.0, 100.0]}',
        ... ))
        >>> limits
        {'plant/a': (0.0, 100.0)}

    """
    threshold = (
        model_params.threshold
        if model_params.threshold is not None
        else 0.99735
    )

    def period_to_timedelta(
        period: str | dt.timedelta | pd.Timedelta,
    ) -> dt.timedelta:
        """Convert a period to a timedelta.

        Args:
            period: Timedelta convertible period.

        Raises:
            ValueError: If unsupported type provided.

        Returns:
            dt.timedelta: Converted period.

        """
        if not isinstance(period, dt.timedelta):
            if isinstance(period, str):
                period = pd.Timedelta(period).to_pytimedelta()
            elif isinstance(period, pd.Timedelta):
                period = period.to_pytimedelta()
        elif isinstance(period, dt.timedelta):
            pass
        else:
            msg = "period must be a timedelta or convertible."
            raise TypeError(msg)
        return period

    if model_params.t_e is None:
        msg = "t_e cannot be None"
        raise ValueError(msg)
    t_e = period_to_timedelta(model_params.t_e)
    t_a_raw = model_params.t_a if model_params.t_a is not None else t_e
    t_a = period_to_timedelta(t_a_raw)
    t_g_raw = model_params.t_g if model_params.t_g is not None else t_e
    t_g = period_to_timedelta(t_g_raw)
    physical_limits = _parse_physical_limits(model_params.physical_limits)
    return threshold, t_e, t_a, t_g, physical_limits


def print_summary(df: pd.DataFrame) -> None:
    """Print a summary of the given DataFrame.

    The function calculates and prints the proportion of anomalous samples
    and the total number of anomalous events based on the 'anomaly' column
    in the DataFrame.

    Args:
        df (DataFrame): The input DataFrame.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'anomaly': [False, True, True, False]})
        >>> print_summary(df)

    """
    text = (
        f"Proportion of anomalous samples: "
        f"{sum(df['anomaly']) / len(df['anomaly']) * 100:.02f}%\n"
        f"Total number of anomalous events: "
        f"{sum(pd.Series(df['anomaly']).diff().dropna() == 1)}"
    )
    logger.info("%s", text)


class RpcOutlierDetector:
    """Streaming outlier detector that processes data from various sources."""

    def __init__(self) -> None:
        """Initialize the detector in a stopped state."""
        self.stopped = True
        # Raw broker/source node captured by get_source so run() can
        # poll its ``stopped`` flag regardless of how the pipeline wraps
        # the source (e.g. the MQTT accumulate/filter chain).
        self._raw_source: Stream | None = None

    def preprocess(
        self,
        x: pd.Series
        | tuple[pd.Timestamp, pd.Series]
        | dict[str, float | str | bytes]
        | MQTTMessage
        | bytes,
        topics: list,
    ) -> dict | None:
        """Normalize heterogeneous input into a ``{time, data}`` dictionary.

        Accepts a pd.Series (with optional Timestamp name), a (timestamp,
        Series) tuple, a plain dict, an MQTTMessage, or raw bytes.

        Args:
            x: Input sample in one of the supported formats.
            topics: Feature names to extract from the input.

        Returns:
            dict | None: Normalized record with keys ``time`` and ``data``,
            or ``None`` when the input type is unrecognized or a value
            cannot be parsed as a float.

        """
        result: dict | None = None
        if isinstance(x, pd.Series):
            if isinstance(x.name, pd.Timestamp):
                t = x.name.tz_localize(None)
            else:
                t = pd.Timestamp.utcnow().tz_localize(None)
            result = {"time": t, "data": x[topics].to_dict()}
        elif isinstance(x, tuple) and isinstance(x[1], (pd.Series)):
            result = {
                "time": cast("pd.Timestamp", x[0]).tz_localize(None),
                "data": x[1][topics].to_dict(),
            }
        else:
            # Timestamps are tz-naive UTC: the rolling model and the
            # consumer's strptime format both reject tz-aware values.
            try:
                if isinstance(x, dict):
                    result = {
                        "time": dt.datetime.now(dt.UTC).replace(
                            microsecond=0,
                            tzinfo=None,
                        ),
                        "data": {
                            k: float(cast("float | str | bytes", v))
                            for k, v in x.items()
                            if k in topics
                        },
                    }
                elif isinstance(x, MQTTMessage):
                    result = {
                        "time": dt.datetime.fromtimestamp(
                            x.timestamp,
                            tz=dt.UTC,
                        ).replace(microsecond=0, tzinfo=None),
                        "data": {x.topic.split("/")[-1]: float(x.payload)},
                    }
                elif isinstance(x, bytes):
                    result = {
                        "time": dt.datetime.now(dt.UTC).replace(
                            microsecond=0,
                            tzinfo=None,
                        ),
                        "data": {topics[0]: float(x.decode("utf-8"))},
                    }
            except (ValueError, TypeError):
                logger.warning("Skipping unparsable message: %r", x)
                result = None
        return result

    def fit_transform(self, x: dict, model: GaussianScorer) -> dict:
        """Apply the anomaly detection model and return a serialisable result.

        Calls ``model.process_one`` with the appropriately shaped feature
        vector, then packages the anomaly flag, adaptive thresholds, and
        optional root-cause feature into a dict with a string timestamp.

        Args:
            x: Preprocessed record with ``time`` and ``data`` keys.
            model: Fitted GaussianScorer or ConditionalGaussianScorer.

        Returns:
            dict: Keys ``time``, ``anomaly``, ``root_cause``, ``level_high``,
            and ``level_low``.

        """
        gaussian_inner = getattr(model.gaussian, "obj", model.gaussian)
        if isinstance(gaussian_inner, MultivariateGaussian):
            x_ = x["data"]
        else:
            x_ = next(iter(x["data"].values()))
        is_anomaly, thresh_high, thresh_low = model.process_one(x_, x["time"])
        if isinstance(model, ConditionalGaussianScorer):
            root_cause = model.get_root_cause()
        else:
            root_cause = None
        return {
            "time": str(x["time"]),
            # **x["data"], # Comment out to lessen the size of payload
            "anomaly": is_anomaly,
            "root_cause": root_cause,
            "level_high": thresh_high,
            "level_low": thresh_low,
        }

    def dump_to_file(self, x: dict, f: IO[str]) -> None:
        """Serialize a result dictionary as JSON and append it to a file.

        Flushes after every line so consumers tailing the file see each
        result as it is produced and nothing is lost on SIGTERM.
        """
        print(json.dumps(x), file=f, flush=True)

    def send_anomaly_email(
        self,
        xs: tuple[dict, dict],
        email_client: EmailClient,
        model: ConditionalGaussianScorer,
    ) -> None:  # pragma: no cover
        """Send an alert email when an anomaly onset is detected.

        Args:
            xs: Sliding window of two consecutive result dicts.
            email_client: Configured email client for sending alerts.
            model: The scorer used to obtain the root-cause feature.

        """
        if len(xs) == 2 and xs[1]["anomaly"] - xs[0]["anomaly"] == 1:
            email_client.send_email(
                f"AID Alert: Anomaly detected in {model.get_root_cause()}",
                xs[1],
            )

    def _warn_on_param_mismatch(
        self,
        model: GaussianScorer,
        threshold: float,
        t_e: dt.timedelta,
        t_a: dt.timedelta,
        t_g: dt.timedelta,
        physical_limits: dict[str, tuple[float, float]]
        | tuple[float, float]
        | None,
    ) -> None:
        """Warn when a recovered model diverges from the configuration.

        A recovery pickle restores the model exactly as saved, so edits
        to ``threshold`` / ``t_e`` / ``t_a`` / ``t_g`` /
        ``physical_limits`` in the config are silently ignored while a
        recovery file exists. Make that visible instead of letting the
        config lie to the operator.
        """
        configured = {
            "threshold": threshold,
            "t_e": t_e,
            "t_a": t_a,
            "grace_period": t_g,
            "physical_limits": physical_limits,
        }
        for name, want in configured.items():
            have = getattr(model, name, None)
            if have != want:
                logger.warning(
                    "Recovered model %s=%r differs from configured %r; "
                    "the recovered value stays in effect. Delete the "
                    "recovery file to apply the new configuration.",
                    name,
                    have,
                    want,
                )

    def _source_file(
        self,
        config: _ClientConfig,
        topics: list,  # noqa: ARG002
        debug: bool,
    ) -> Stream:
        config = cast("FileClient", config)
        if debug:
            source = Stream()
        else:
            data = pd.read_csv(config.path, index_col=0)
            data.index = pd.to_datetime(data.index, utc=True)
            source = Stream.from_iterable(data.iterrows())
        self._raw_source = source
        return source

    def _source_mqtt(
        self,
        config: _ClientConfig,
        topics: list,
        debug: bool,  # noqa: ARG002
    ) -> Stream:
        config = cast("MQTTClient", config)
        source = Stream.from_mqtt(
            **config.model_dump(),
            topic=[(topic, 0) for topic in topics],
        )
        # Capture the raw broker node before wrapping: run() polls
        # its ``stopped`` flag, which the accumulate/filter nodes
        # below do not expose.
        self._raw_source = source
        return source.accumulate(
            _func,
            start={},
            topics=topics,
        ).filter(_filt, topics)

    def _source_kafka(
        self,
        config: _ClientConfig,
        topics: list,
        debug: bool,  # noqa: ARG002
    ) -> Stream:
        config = cast("KafkaClient", config)
        # "detection_service" is only a default: a user-supplied
        # group.id in the config must win.
        source = Stream.from_kafka(
            topics,
            {"group.id": "detection_service", **config.model_dump()},
        )
        self._raw_source = source
        return source

    def _source_pulsar(
        self,
        config: _ClientConfig,
        topics: list,
        debug: bool,  # noqa: ARG002
    ) -> Stream:
        config = cast("PulsarClient", config)
        if not _PULSAR_AVAILABLE:
            msg = "pulsar-client is not installed"
            raise RuntimeError(msg)
        source = Stream.from_pulsar(
            config.service_url,
            topics,
            subscription_name="detection_service",
        )
        self._raw_source = source
        return source

    def _source_nats(
        self,
        config: _ClientConfig,
        topics: list,
        debug: bool,  # noqa: ARG002
    ) -> Stream:
        config = cast("NATSClient", config)
        source = Stream.from_nats(
            servers=config.servers,
            topic=topics,
        )
        # Capture the raw broker node before wrapping: run() polls
        # its ``stopped`` flag, which the accumulate/filter nodes
        # below do not expose. The NATSMessage adapter exposes the
        # same ``.topic``/``.payload`` interface as MQTTMessage, so
        # the accumulate/_func/filter chain is identical to MQTT.
        self._raw_source = source
        return source.accumulate(
            _func,
            start={},
            topics=topics,
        ).filter(_filt, topics)

    def get_source(
        self,
        config: _ClientConfig,
        topics: list,
        debug: bool = False,
    ) -> Stream:
        """Return a Streamz source stream based on the transport configuration.

        Dispatches to ``from_iterable`` (file), ``from_mqtt``, ``from_kafka``,
        or ``from_pulsar`` depending on the keys present in ``config``.

        Args:
            config: Client configuration dict identifying the transport type.
            topics: Feature or subscription topic names.
            debug: When True and config is a FileClient, return a bare Stream
                for manual event injection.

        Returns:
            streamz.Stream: Configured source stream.

        Raises:
            RuntimeError: If the Pulsar transport is requested but
                pulsar-client is not installed, or if no recognised
                transport key is found in config.

        """
        entry = _TRANSPORT_REGISTRY.get(type(config))
        if entry is None:
            # Unrecognised transport: a runtime configuration error, not a
            # static type error, so RuntimeError (as before) is preserved.
            msg = f"Wrong client: {config}"
            raise RuntimeError(msg)
        source_handler, _ = entry
        return source_handler(self, config, topics, debug)

    def _sink_file(
        self,
        config: _ClientConfig,
        detector: Stream,
        prefix: str,  # noqa: ARG002
        topic: str,  # noqa: ARG002
    ) -> None:
        config = cast("FileClient", config)
        output_path = Path(config.output)
        f = output_path.open("a")
        _exit_stack.callback(f.close)
        detector.sink(self.dump_to_file, f)

    def _sink_mqtt(
        self,
        config: _ClientConfig,
        detector: Stream,
        prefix: str,
        topic: str,  # noqa: ARG002
    ) -> None:
        config = cast("MQTTClient", config)
        detector.to_mqtt(
            **config.model_dump(),
            topic=prefix,
            publish_kwargs={"retain": True},
        )

    def _sink_kafka(
        self,
        config: _ClientConfig,
        detector: Stream,
        prefix: str,  # noqa: ARG002
        topic: str,
    ) -> None:
        config = cast("KafkaClient", config)
        detector.map(lambda x: (str(x), "dynamic_limits")).to_kafka(
            topic,
            config.model_dump(),
        )

    def _sink_pulsar(
        self,
        config: _ClientConfig,
        detector: Stream,
        prefix: str,  # noqa: ARG002
        topic: str,
    ) -> None:
        config = cast("PulsarClient", config)
        if not _PULSAR_AVAILABLE:
            msg = "pulsar-client is not installed"
            raise RuntimeError(msg)

        class Example(Record):  # type: ignore[misc]
            time = PulsarString()
            anomaly = PulsarString()
            level_high = PulsarString()
            level_low = PulsarString()

        detector.map(lambda x: Example(**x)).to_pulsar(
            config.service_url,
            topic,
            producer_config={"schema": JsonSchema(Example)},
        )

    def _sink_nats(
        self,
        config: _ClientConfig,
        detector: Stream,
        prefix: str,
        topic: str,  # noqa: ARG002
    ) -> None:
        config = cast("NATSClient", config)
        detector.to_nats(
            servers=config.servers,
            topic=prefix,
        )

    def get_sink(
        self,
        config: _ClientConfig,
        topics: list,
        detector: Stream,
        out_topics: list[str] | str | None = None,
    ) -> Stream:
        """Get the data sink based on the provided configuration.

        Args:
            config (dict): The configuration dictionary.
            topics (list): The input topics the detector subscribes to.
            detector (streamz.core.map): Upstream streamz pipeline.
            out_topics (list | str | None): Configured output topic names.
                When provided, the first entry names the sink topic and the
                common prefix of all entries is used as the MQTT topic
                prefix; otherwise both are derived from ``topics``.

        Returns:
            streamz.core.map: streamz pipeline with sink

        """
        if isinstance(out_topics, str):
            out_topics = [out_topics] if out_topics else []
        out_topics_: list[str] = [t for t in out_topics or [] if t]
        if out_topics_:
            prefix: str = common_prefix(list(out_topics_))
            topic: str = out_topics_[0]
        else:
            prefix = common_prefix(topics)
            topic = f"{prefix}dynamic_limits"
        logger.info("Sinking to '%s'\n", topic)
        entry = _TRANSPORT_REGISTRY.get(type(config))
        if entry is not None:
            _, sink_handler = entry
            sink_handler(self, config, detector, prefix, topic)
        return detector

    def run(
        self,
        config: _ClientConfig,
        source: Stream,
        detector: Stream,
        debug: bool,
    ) -> None:
        """Run the detection pipeline until the source stream is exhausted.

        Args:
            config: Client configuration used to determine debug file paths.
            source: Streamz source stream.
            detector: Streamz pipeline terminating at a sink.
            debug: When True, replay a small CSV batch instead of streaming.
                Only valid with a file client; ``start`` rejects the
                combination of debug mode and a remote broker upfront.

        """
        if debug and isinstance(config, FileClient):
            logger.info("=== Debugging started... ===")
            data = pd.read_csv(config.path, index_col=0)
            data.index = pd.to_datetime(data.index, utc=True)
            for row in data.head().iterrows():
                source.emit(row)
            logger.info("=== Debugging finished with success... ===")
        else:
            detector.start()
            logger.info("=== Service started ===")

            # Poll the raw source node captured by get_source rather
            # than probing a hardcoded upstream depth of the pipeline.
            raw_source = (
                self._raw_source if self._raw_source is not None else source
            )
            while not raw_source.stopped:
                time.sleep(2)  # pragma: no cover

    def start(
        self,
        client: _ClientConfig,
        io: IOConfig,
        model_params: ModelConfig,
        setup: SetupConfig,
        email: EmailConfig | None = None,
    ) -> None:
        """Set up and run the streaming anomaly detection pipeline.

        Creates a GaussianScorer (or ConditionalGaussianScorer for multivariate
        data), wires together the source, detection, optional encryption, and
        sink stages, then delegates execution to ``run``.

        Args:
            client: Transport configuration (file, MQTT, Kafka, or Pulsar).
            io: I/O configuration with ``in_topics`` and ``out_topics``.
            model_params: Model hyper-parameters including ``t_e`` and optional
                ``t_a``, ``t_g``, ``threshold``, and ``physical_limits``
                (static per-signal operating bounds).
            setup: Runtime options such as ``debug``, ``key_path``, and
                ``recovery_path``.
            email: Optional email alert configuration.

        Raises:
            ValueError: If debug mode is requested with a remote broker
                configuration; debug replays a CSV file and therefore
                requires a file client.

        """
        recovery_path = setup.recovery_path or ""
        key_path = setup.key_path or ""
        debug = bool(setup.debug)
        if debug and not isinstance(client, FileClient):
            msg = (
                "Debug mode replays a CSV file and requires a file "
                "client; got a remote broker configuration instead."
            )
            raise ValueError(msg)

        in_topics = io.in_topics
        out_topics = io.out_topics

        threshold, t_e, t_a, t_g, physical_limits = expand_model_params(
            model_params,
        )
        # The univariate scorer takes the bounds of its single signal;
        # the conditional scorer keeps the whole per-feature mapping.
        univariate_limits = (
            physical_limits.get(in_topics[0])
            if physical_limits and in_topics
            else None
        )
        scoped_limits: (
            dict[str, tuple[float, float]] | tuple[float, float] | None
        ) = physical_limits if len(in_topics) > 1 else univariate_limits

        model = load_model(recovery_path, in_topics)

        if model is not None:
            self._warn_on_param_mismatch(
                model,
                threshold,
                t_e,
                t_a,
                t_g,
                scoped_limits,
            )
        if model is None:
            if len(in_topics) > 1:
                obj = MultivariateGaussian()
                model = ConditionalGaussianScorer(
                    utils.TimeRolling(obj, period=t_e),
                    threshold=threshold,
                    grace_period=t_g,
                    t_a=t_a,
                    physical_limits=physical_limits,
                )
            else:
                obj = proba.Gaussian()
                model = GaussianScorer(
                    utils.TimeRolling(obj, period=t_e),
                    threshold=threshold,
                    grace_period=t_g,
                    t_a=t_a,
                    physical_limits=univariate_limits,
                )

        source = self.get_source(client, in_topics, debug)

        detector = (
            source.map(self.preprocess, in_topics)
            .filter(lambda x: x is not None)
            .map(self.fit_transform, model)
        )
        # Email alerting branches off the plaintext detector node; after
        # the sign/encrypt maps the anomaly flags are opaque strings.
        plain = detector

        if key_path:
            sender, _ = init_rsa_security(key_path)
            # Encrypt-then-sign-over-ciphertext (format version 2): the
            # consumer verifies the signature before decrypting.
            detector = detector.map(encrypt_and_sign_data, sender)
        detector = self.get_sink(client, in_topics, detector, out_topics)
        if email is not None and email.sender_email is not None:
            email_client = EmailClient(**email.model_dump())
            plain.sliding_window(2).sink(
                self.send_anomaly_email,
                email_client,
                model,
            )

        try:
            self.run(client, source, detector, debug)
        finally:
            detector.stop()
            # Close (and thereby flush) any files opened by the sink on
            # every shutdown path, not only in debug mode.
            _exit_stack.close()
            logger.info("=== Service stopped ===")
            save_model(recovery_path, in_topics, model)


# Single transport dispatch table: maps each Pydantic client model to its
# (source_factory, sink_factory) handler pair. get_source/get_sink look the
# config type up here instead of walking an isinstance if/elif chain, so a
# new transport is added by registering one entry. An unknown type yields
# no entry, which get_source surfaces as the historical
# RuntimeError("Wrong client: ...").
_SourceHandler = Callable[
    [RpcOutlierDetector, _ClientConfig, list, bool],
    Stream,
]
_SinkHandler = Callable[
    [RpcOutlierDetector, _ClientConfig, Stream, str, str],
    None,
]
_TRANSPORT_REGISTRY: dict[
    type,
    tuple[_SourceHandler, _SinkHandler],
] = {
    FileClient: (
        RpcOutlierDetector._source_file,
        RpcOutlierDetector._sink_file,
    ),
    MQTTClient: (
        RpcOutlierDetector._source_mqtt,
        RpcOutlierDetector._sink_mqtt,
    ),
    KafkaClient: (
        RpcOutlierDetector._source_kafka,
        RpcOutlierDetector._sink_kafka,
    ),
    PulsarClient: (
        RpcOutlierDetector._source_pulsar,
        RpcOutlierDetector._sink_pulsar,
    ),
    NATSClient: (
        RpcOutlierDetector._source_nats,
        RpcOutlierDetector._sink_nats,
    ),
}
