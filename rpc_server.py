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
            dict: Normalized record with keys ``time`` and ``data``.

        """
        if isinstance(x, pd.Series):
            if isinstance(x.name, pd.Timestamp):
                t = x.name.tz_localize(None)
            else:
                t = pd.Timestamp.utcnow().tz_localize(None)
            return {"time": t, "data": x[topics].to_dict()}
        if isinstance(x, tuple) and isinstance(x[1], (pd.Series)):
            return {
                "time": cast("pd.Timestamp", x[0]).tz_localize(None),
                "data": x[1][topics].to_dict(),
            }
        if isinstance(x, dict):
            return {
                "time": dt.datetime.now(dt.UTC).replace(microsecond=0),
                "data": {
                    k: float(cast("float | str | bytes", v))
                    for k, v in x.items()
                    if k in topics
                },
            }
        if isinstance(x, MQTTMessage):
            return {
                "time": dt.datetime.fromtimestamp(
                    x.timestamp,
                    tz=dt.UTC,
                ).replace(microsecond=0),
                "data": {x.topic.split("/")[-1]: float(x.payload)},
            }
        if isinstance(x, bytes):
            return {
                "time": dt.datetime.now(dt.UTC).replace(microsecond=0),
                "data": {topics[0]: float(x.decode("utf-8"))},
            }
        return None

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

    def dump_to_file(self, x: dict, f: IO[str]) -> None:  # pragma: no cover
        """Serialize a result dictionary as JSON and append it to a file."""
        print(json.dumps(x), file=f)

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
            RuntimeError: If no recognised transport key is found in config.

        """
        if istypedinstance(config, FileClient):
            if debug:
                source = Stream()
            else:
                data = pd.read_csv(config.get("path", ""), index_col=0)
                data.index = pd.to_datetime(data.index, utc=True)
                source = Stream.from_iterable(data.iterrows())
        elif istypedinstance(config, MQTTClient):
            source = Stream.from_mqtt(
                **config,
                topic=[(topic, 0) for topic in topics],
            )
            source = source.accumulate(
                _func,
                start={},
                topics=topics,
            ).filter(_filt, topics)
        elif istypedinstance(config, KafkaClient):
            source = Stream.from_kafka(
                topics,
                {**config, "group.id": "detection_service"},
            )
        elif istypedinstance(config, PulsarClient):
            msg = "Pulsar client requires Python < 3.12.*"
            raise ValueError(msg)
        else:
            msg = f"Wrong client: {config}"
            raise RuntimeError(msg)
        return source

    def get_sink(
        self,
        config: FileClient | MQTTClient | KafkaClient | PulsarClient,
        topics: list,
        detector: Stream,
    ) -> Stream:
        """Get the data sink based on the provided configuration.

        Args:
            config (dict): The configuration dictionary.
            topics (list): The topics to subscribe to.
            detector (streamz.core.map): Upstream streamz pipeline.

        Returns:
            streamz.core.map: streamz pipeline with sink

        """
        prefix: str = common_prefix(topics)
        topic: str = f"{prefix}dynamic_limits"
        logger.info("Sinking to '%s'\n", topic)
        if istypedinstance(config, FileClient):
            output_path = Path(config.get("output", ""))
            f = output_path.open("a")
            _exit_stack.callback(f.close)
            detector.sink(self.dump_to_file, f)
        elif istypedinstance(config, MQTTClient):  # pragma: no cover
            detector.to_mqtt(
                **config,
                topic=prefix,
                publish_kwargs={"retain": True},
            )
        # TODO(MarekWadinger): add coverage test
        elif istypedinstance(config, KafkaClient):  # pragma: no cover
            detector.map(lambda x: (str(x), "dynamic_limits")).to_kafka(
                topic,
                config,
            )
        elif istypedinstance(config, PulsarClient):  # pragma: no cover
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

        """
        # TODO(MarekWadinger): handle combination of debug and remote broker
        if debug and istypedinstance(config, FileClient):
            logger.info("=== Debugging started... ===")
            data = pd.read_csv(cast("FileClient", config)["path"], index_col=0)
            data.index = pd.to_datetime(data.index, utc=True)
            for row in data.head().iterrows():
                source.emit(row)
            _exit_stack.close()
            logger.info("=== Debugging finished with success... ===")
        else:  # pragma: no cover
            detector.start()
            logger.info("=== Service started ===")

            while True:
                try:
                    if source.stopped:
                        break
                except AttributeError:
                    if source.upstreams[0].upstreams[0].stopped:
                        break
                time.sleep(2)

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

        """
        recovery_path = setup.get("recovery_path", "")
        key_path = setup.get("key_path", "")
        debug = setup.get("debug", False)

        in_topics = io.get("in_topics", [])
        # TODO(MarekWadinger): use out_topics
        _ = io.get("out_topics", None)

        threshold, t_e, t_a, t_g = expand_model_params(model_params)

        model = load_model(recovery_path, in_topics)

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

        detector = source.map(self.preprocess, in_topics).map(
            self.fit_transform,
            model,
        )

        if key_path:
            sender, _ = init_rsa_security(key_path)
            detector = (
                detector.map(sign_data, sender)
                .map(encrypt_data, sender)
                .map(decode_data)
            )
        detector = self.get_sink(client, in_topics, detector)
        if email is not None and email.get("sender_email") is not None:
            email_client = EmailClient(**email)
            detector.sliding_window(2).sink(
                self.send_anomaly_email,
                email_client,
                model,
            )

        try:
            self.run(client, source, detector, debug)
        finally:
            detector.stop()
            logger.info("=== Service stopped ===")
            save_model(recovery_path, in_topics, model)
