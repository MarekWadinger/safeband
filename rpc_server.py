"""Streaming anomaly detection server with MQTT, Kafka, and file I/O."""

# IMPORTS
import contextlib
import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import IO, cast

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

from functions.anomaly import ConditionalGaussianScorer, GaussianScorer
from functions.email_client import EmailClient
from functions.encryption import (
    decode_data,
    encrypt_data,
    init_rsa_security,
    sign_data,
)
from functions.model_persistence import load_model, save_model
from functions.proba import MultivariateGaussian
from functions.streamz_tools import _filt, _func, to_mqtt  # noqa: F401
from functions.typing_extras import (
    EmailConfig,
    FileClient,
    IOConfig,
    KafkaClient,
    ModelConfig,
    MQTTClient,
    PulsarClient,
    SetupConfig,
    istypedinstance,
)
from functions.utils import common_prefix

logger = logging.getLogger(__name__)

# CONSTANTS

_exit_stack: contextlib.ExitStack = contextlib.ExitStack()


# DEFINITIONS
def expand_model_params(
    model_params: ModelConfig,
) -> tuple[float, dt.timedelta, dt.timedelta, dt.timedelta]:
    """Extract and convert model parameters from the configuration dictionary.

    Args:
        model_params: Mapping containing threshold, t_e, t_a, and t_g values.

    Returns:
        tuple: ``(threshold, t_e, t_a, t_g)`` as a float and three timedeltas.

    """
    threshold = model_params.get("threshold", 0.99735)

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

    t_e = model_params.get("t_e")
    if t_e is None:
        msg = "t_e cannot be None"
        raise ValueError(msg)
    t_e = period_to_timedelta(t_e)
    t_a = cast("pd.Timedelta", model_params.get("t_a", t_e))
    t_a = period_to_timedelta(t_a)
    t_g = cast("pd.Timedelta", model_params.get("t_g", t_e))
    t_g = period_to_timedelta(t_g)
    return threshold, t_e, t_a, t_g


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
    ) -> None:
        """Warn when a recovered model diverges from the configuration.

        A recovery pickle restores the model exactly as saved, so edits
        to ``threshold`` / ``t_e`` / ``t_a`` / ``t_g`` in the config are
        silently ignored while a recovery file exists. Make that
        visible instead of letting the config lie to the operator.
        """
        configured = {
            "threshold": threshold,
            "t_e": t_e,
            "t_a": t_a,
            "grace_period": t_g,
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

    def get_source(
        self,
        config: FileClient | MQTTClient | KafkaClient | PulsarClient,
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
        if istypedinstance(config, FileClient):
            if debug:
                source = Stream()
            else:
                data = pd.read_csv(config.get("path", ""), index_col=0)
                data.index = pd.to_datetime(data.index, utc=True)
                source = Stream.from_iterable(data.iterrows())
            self._raw_source = source
        elif istypedinstance(config, MQTTClient):
            source = Stream.from_mqtt(
                **config,
                topic=[(topic, 0) for topic in topics],
            )
            # Capture the raw broker node before wrapping: run() polls
            # its ``stopped`` flag, which the accumulate/filter nodes
            # below do not expose.
            self._raw_source = source
            source = source.accumulate(
                _func,
                start={},
                topics=topics,
            ).filter(_filt, topics)
        elif istypedinstance(config, KafkaClient):
            # "detection_service" is only a default: a user-supplied
            # group.id in the config must win.
            source = Stream.from_kafka(
                topics,
                {"group.id": "detection_service", **config},
            )
            self._raw_source = source
        elif istypedinstance(config, PulsarClient):
            if not _PULSAR_AVAILABLE:
                msg = "pulsar-client is not installed"
                raise RuntimeError(msg)
            source = Stream.from_pulsar(
                config.get("service_url"),
                topics,
                subscription_name="detection_service",
            )
            self._raw_source = source
        else:
            msg = f"Wrong client: {config}"
            raise RuntimeError(msg)
        return source

    def get_sink(
        self,
        config: FileClient | MQTTClient | KafkaClient | PulsarClient,
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
        if istypedinstance(config, FileClient):
            output_path = Path(config.get("output", ""))
            f = output_path.open("a")
            _exit_stack.callback(f.close)
            detector.sink(self.dump_to_file, f)
        elif istypedinstance(config, MQTTClient):
            detector.to_mqtt(
                **config,
                topic=prefix,
                publish_kwargs={"retain": True},
            )
        elif istypedinstance(config, KafkaClient):
            detector.map(lambda x: (str(x), "dynamic_limits")).to_kafka(
                topic,
                config,
            )
        elif istypedinstance(config, PulsarClient):
            if not _PULSAR_AVAILABLE:
                msg = "pulsar-client is not installed"
                raise RuntimeError(msg)

            class Example(Record):  # type: ignore[misc]
                time = PulsarString()
                anomaly = PulsarString()
                level_high = PulsarString()
                level_low = PulsarString()

            detector.map(lambda x: Example(**x)).to_pulsar(
                config.get("service_url"),
                topic,
                producer_config={"schema": JsonSchema(Example)},
            )

        return detector

    def run(
        self,
        config: FileClient | MQTTClient | KafkaClient | PulsarClient,
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
        if debug and istypedinstance(config, FileClient):
            logger.info("=== Debugging started... ===")
            data = pd.read_csv(cast("FileClient", config)["path"], index_col=0)
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
        client: FileClient | MQTTClient | KafkaClient | PulsarClient,
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
                ``t_a``, ``t_g``, and ``threshold``.
            setup: Runtime options such as ``debug``, ``key_path``, and
                ``recovery_path``.
            email: Optional email alert configuration.

        Raises:
            ValueError: If debug mode is requested with a remote broker
                configuration; debug replays a CSV file and therefore
                requires a file client.

        """
        recovery_path = setup.get("recovery_path", "")
        key_path = setup.get("key_path", "")
        debug = setup.get("debug", False)
        if debug and not istypedinstance(client, FileClient):
            msg = (
                "Debug mode replays a CSV file and requires a file "
                "client; got a remote broker configuration instead."
            )
            raise ValueError(msg)

        in_topics = io.get("in_topics", [])
        out_topics = io.get("out_topics", None)

        threshold, t_e, t_a, t_g = expand_model_params(model_params)

        model = load_model(recovery_path, in_topics)

        if model is not None:
            self._warn_on_param_mismatch(model, threshold, t_e, t_a, t_g)
        if model is None:
            if len(in_topics) > 1:
                obj = MultivariateGaussian()
                model = ConditionalGaussianScorer(
                    utils.TimeRolling(obj, period=t_e),
                    threshold=threshold,
                    grace_period=t_g,
                    t_a=t_a,
                )
            else:
                obj = proba.Gaussian()
                model = GaussianScorer(
                    utils.TimeRolling(obj, period=t_e),
                    threshold=threshold,
                    grace_period=t_g,
                    t_a=t_a,
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
            detector = (
                detector.map(sign_data, sender)
                .map(encrypt_data, sender)
                .map(decode_data)
            )
        detector = self.get_sink(client, in_topics, detector, out_topics)
        if email is not None and email.get("sender_email") is not None:
            email_client = EmailClient(**email)
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
