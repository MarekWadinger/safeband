# IMPORTS
import datetime as dt
import json
import time
from typing import IO, Union

import pandas as pd
from paho.mqtt.client import MQTTMessage
from river import proba, utils
from streamz import Stream

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

# CONSTANTS
GRACE_PERIOD = 60*2
WINDOW = dt.timedelta(hours=24*1)

open_files: list[IO] = []


# DEFINITIONS
def expand_model_params(model_params):
    threshold = model_params.get("threshold", 0.99735)

    def period_to_timedelta(
            period: Union[str, dt.timedelta, pd.Timedelta]) -> dt.timedelta:
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
            raise ValueError("period must be a timedelta or convertible.")
        return period

    t_e = model_params.get("t_e")
    if t_e is None:
        raise ValueError("t_e cannot be None")
    t_e = period_to_timedelta(t_e)
    t_a = model_params.get("t_a", t_e)
    t_a = period_to_timedelta(t_a)
    t_g = model_params.get("t_g", t_e)
    t_g = period_to_timedelta(t_g)
    return threshold, t_e, t_a, t_g


def print_summary(df):
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
        Proportion of anomalous samples: 50.00%
        Total number of anomalous events: 2
    """
    text = (
        f"Proportion of anomalous samples: "
        f"{sum(df['anomaly'])/len(df['anomaly'])*100:.02f}%\n"
        f"Total number of anomalous events: "
        f"{sum(pd.Series(df['anomaly']).diff().dropna() == 1)}")
    print(text)


class RpcOutlierDetector:
    def __init__(self):
        self.stopped = True

    def preprocess(
            self,
            x,
            topics: list):
        """Preprocess the input data.

        Args:
            x (Union[pd.Series, tuple, dict, MQTTMessage, bytes]): The input
            data
            to be preprocessed.
            topics (list): The topics to be extracted from the
            input data.

        Returns:
            dict: The preprocessed data.

        Examples:
        >>> series = pd.Series([1.], name=pd.to_datetime('2023-01-01'),
        ...                    index=["sensor_1"])
        >>> obj = RpcOutlierDetector()
        >>> obj.preprocess(series, ['sensor_1'])
        {'time': Timestamp('2023-01-01 00:00:00'), 'data': {'sensor_1': 1.0}}

        >>> series_tuple = (pd.to_datetime('2023-01-01'), series)
        >>> obj.preprocess(series_tuple, ['sensor_1'])
        {'time': Timestamp('2023-01-01 00:00:00'), 'data': {'sensor_1': 1.0}}

        >>> data_dict = {'time': pd.to_datetime('2023-01-01'), 'sensor_1': 1.}
        >>> out = obj.preprocess(data_dict, ['sensor_1'])
        >>> out.keys(), out['data'].keys()
        (dict_keys(['time', 'data']), dict_keys(['sensor_1']))

        >>> mqtt_message = MQTTMessage()
        >>> mqtt_message.timestamp = 1672527600.0
        >>> mqtt_message.payload = b'1.'
        >>> mqtt_message.topic = b'sensors/sensor_1'
        >>> out = obj.preprocess(mqtt_message, ['1'])
        >>> out.keys(), out['data'].keys()
        (dict_keys(['time', 'data']), dict_keys(['sensor_1']))

        >>> binary_data = b'1.0'
        >>> out = obj.preprocess(binary_data, ['sensor_1'])
        >>> out.keys(), out['data'].keys()
        (dict_keys(['time', 'data']), dict_keys(['sensor_1']))
        """
        if isinstance(x, pd.Series):
            if isinstance(x.name, pd.Timestamp):
                t = x.name.tz_localize(None)
            else:
                t = pd.Timestamp.utcnow().tz_localize(None)
            return {"time": t,
                    "data": x[topics].to_dict()
                    }
        if isinstance(x, tuple) and isinstance(x[1], (pd.Series)):
            return {"time": x[0].tz_localize(None),
                    "data": x[1][topics].to_dict()
                    }
        if isinstance(x, dict):
            return {"time": dt.datetime.utcnow().replace(microsecond=0),
                    "data": {k: float(v) for k, v in x.items() if k in topics}
                    }
        if isinstance(x, MQTTMessage):
            return {"time": dt.datetime
                    .fromtimestamp(x.timestamp).replace(microsecond=0),
                    "data": {x.topic.split("/")[-1]: float(x.payload)}
                    }
        if isinstance(x, bytes):
            return {"time": dt.datetime.utcnow().replace(microsecond=0),
                    "data": {topics[0]: float(x.decode("utf-8"))}
                    }

    def fit_transform(
            self,
            x,
            model: GaussianScorer):
        """Apply anomaly detection model to the input data.

        The function applies the provided anomaly detection model to the input
        data and returns the result as a dictionary.

        Args:
            x (dict): The input data dictionary.
            model: The anomaly detection model.

        Returns:
            dict: The processed data dictionary.

        Examples:
        >>> x = {"time": dt.datetime(2022,1,1),
        ...      "data": {"feature1": 0.5, "feature2": 1.2, "feature3": -0.8}}
        >>> model = model = GaussianScorer(
        ...     utils.TimeRolling(proba.Gaussian(), period=WINDOW),
        ...     grace_period=GRACE_PERIOD)
        >>> obj = RpcOutlierDetector()
        >>> result = obj.fit_transform(x, model)
        >>> sorted(result.keys())
        ['anomaly', 'level_high', 'level_low', 'time']
        >>> isinstance(result["time"], str)
        True
        >>> isinstance(result["anomaly"], int)
        True
        >>> isinstance(result["level_high"], float)
        True
        >>> isinstance(result["level_low"], float)
        True
        """
        if isinstance(model.gaussian.obj, MultivariateGaussian):
            x_ = x["data"]
        else:
            x_ = next(iter(x["data"].values()))
        is_anomaly, thresh_high, thresh_low = model.process_one(x_, x["time"])
        return {"time": str(x["time"]),
                # **x["data"], # Comment out to lessen the size of payload
                "anomaly": is_anomaly,
                "level_high": thresh_high,
                "level_low": thresh_low
                }

    def dump_to_file(self, x, f):  # pragma: no cover
        print(json.dumps(x), file=f)

    def send_anomaly_email(
            self,
            xs: tuple[dict, dict],
            email_client: EmailClient):  # pragma: no cover
        if len(xs) == 2 and xs[1]["anomaly"] - xs[0]["anomaly"] == 1:
            email_client.send_email(xs[1])

    def get_source(
            self,
            config: Union[FileClient, MQTTClient, KafkaClient, PulsarClient],
            topics: list,
            debug: bool = False):
        """Get the data source based on the provided configuration.

        The function returns a data source stream object based on the
        configuration settings.
        If the 'path' key is present in the config, it returns a stream from an
        iterable of
        rows in the 'data' dictionary. If the 'host' key is present, it
        returns a stream from MQTT messages with the specified topics. If the
        'bootstrap.servers' key is present, it returns a stream from Kafka
        messages with the specified topics. If none of the expected keys are
        found, it raises a RuntimeError.

        Args:
            config (dict): The configuration dictionary.
            topics (list): The topics to subscribe to for MQTT or Kafka sources.
            debug (bool, optional): Enable debug mode. Defaults to False.

        Returns:
            stream.Stream: The data source stream object.

        Raises:
            RuntimeError: If the data format is incorrect.

        Examples:
        >>> config = {
        ...     "path": "tests/test.csv",
        ...     "output": "tests/output.json"}
        >>> topics = ["test"]
        >>> obj = RpcOutlierDetector()
        >>> source = obj.get_source(config, topics)
        >>> type(source)
        <class 'streamz.sources.from_iterable'>

        >>> source = obj.get_source(config, topics, debug=True)
        >>> type(source)
        <class 'streamz.core.Stream'>

        >>> config = {"host": "mqtt.server", "port": 1883}
        >>> topics = ["test"]
        >>> source = obj.get_source(config, topics)
        >>> type(source)
        <class 'streamz.core.filter'>

        >>> config = {"bootstrap_servers": "kafka.server:9092",
        ...           "group.id": "consumer-group"}
        >>> topics = ["kafka-topics"]
        >>> source = obj.get_source(config, topics)
        >>> type(source)
        <class 'streamz.sources.from_kafka'>

        # >>> config = {"service_url": "pulsar://localhost:6650"}
        # >>> topics = ["pulsar-topics"]
        # >>> source = obj.get_source(config, topics)
        # >>> type(source)
        # <class 'streamz_pulsar.sources.from_pulsar.from_pulsar'>

        >>> config = {"invalid": "config"}
        >>> topics = ["test"]
        >>> source = obj.get_source(config, topics)
        Traceback (most recent call last):
        ...
        RuntimeError: Wrong client.
        """  # noqa: E501
        if istypedinstance(config, FileClient):
            if debug:
                source = Stream()
            else:
                data = pd.read_csv(config.get("path", ""), index_col=0)
                data.index = pd.to_datetime(data.index, utc=True)
                source = Stream.from_iterable(data.iterrows())
        elif istypedinstance(config, MQTTClient):
            source = Stream.from_mqtt(
                **config, topic=[(topic, 0) for topic in topics])
            source = source.accumulate(
                _func, start={}, **{"topics": topics}).filter(_filt, topics)
        elif istypedinstance(config, KafkaClient):
            source = Stream.from_kafka(
                topics, {**config, 'group.id': 'detection_service'})
        elif istypedinstance(config, PulsarClient):
            import sys
            if sys.version_info.major == 3 and sys.version_info.minor < 12:
                source = Stream.from_pulsar(
                    config.get("service_url"),
                    topics,
                    subscription_name='detection_service')
            else:
                raise ValueError("Pulsar client requires Python < 3.12.*")
        else:
            raise RuntimeError(f"Wrong client: {config}")
        return source

    def get_sink(
            self,
            config: Union[FileClient, MQTTClient, KafkaClient, PulsarClient],
            topics: list,
            detector):
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
        print(f"Sinking to '{topic}'\n")
        if istypedinstance(config, FileClient):
            f = open(config.get("output", ""), 'a')
            open_files.append(f)
            detector.sink(self.dump_to_file, f)
        elif istypedinstance(config, MQTTClient):  # pragma: no cover
            detector.to_mqtt(
                **config, topic=prefix, publish_kwargs={"retain": True})
        # TODO: add coverage test
        elif istypedinstance(config, KafkaClient):  # pragma: no cover
            detector.map(lambda x: (str(x), "dynamic_limits")
                         ).to_kafka(topic, config)
        elif istypedinstance(config, PulsarClient):  # pragma: no cover
            from pulsar.schema import JsonSchema, Record, String

            class Example(Record):
                time = String()
                anomaly = String()
                level_high = String()
                level_low = String()
            detector.map(lambda x: Example(**x)).to_pulsar(
                config.get("service_url"),
                topic,
                producer_config={"schema": JsonSchema(Example)})

        return detector

    def run(
            self,
            config,
            source,
            detector,
            debug
    ):
        # TODO: handle combination of debug and remote broker
        if debug and istypedinstance(config, FileClient):
            print("=== Debugging started... ===")
            data = pd.read_csv(config['path'], index_col=0)
            data.index = pd.to_datetime(data.index, utc=True)
            for row in data.head().iterrows():
                source.emit(row)
            for file in open_files:
                file.close()
            print("=== Debugging finished with success... ===")
        else:  # pragma: no cover
            detector.start()
            print("=== Service started ===")

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
            client: Union[FileClient, MQTTClient, KafkaClient, PulsarClient],
            io: IOConfig,
            model_params: ModelConfig,
            setup: SetupConfig,
            email: Union[EmailConfig, None] = None,
    ):
        """Process the limits in a streaming manner.

        The function sets up the necessary components for streaming processing
        of limits. It creates instances of the GaussianScorer model for
        anomaly detection, prepares the data source based on the
        configuration, and performs the required transformations.
        The processed data is then stored or published based on the
        configuration.

        Args:
            config (dict): The configuration dictionary.
            topics (list): The topics to subscribe to for sources.
            key_path (str): The path to the RSA keys
            debug (bool, optional): Enable debug mode. Defaults to False.

        Examples:
        >>> client = {"path": "tests/test.csv", "output": "tests/output.json"}
        >>> io = {"in_topics": ["A"]}
        >>> model_params = {"t_e": "1H"}
        >>> setup = {"key_path": ".temp", "debug": True}
        >>> obj = RpcOutlierDetector()
        >>> obj.start(client, io, model_params, setup)
        Sinking to 'dynamic_limits'
        <BLANKLINE>
        === Debugging started... ===
        === Debugging finished with success... ===
        === Service stopped ===
        """
        recovery_path = setup.get("recovery_path", "")
        key_path = setup.get("key_path", "")
        debug = setup.get("debug", False)

        in_topics = io.get("in_topics", [])
        # TODO: use out_topics
        _ = io.get("out_topics", None)

        threshold, t_e, t_a, t_g = expand_model_params(model_params)

        model = load_model(recovery_path, in_topics)

        if model is None:
            if len(in_topics) > 1:
                obj = MultivariateGaussian()
                model = ConditionalGaussianScorer(
                    utils.TimeRolling(obj, period=t_e),
                    threshold=threshold,
                    grace_period=GRACE_PERIOD)
            else:
                obj = proba.Gaussian()
                model = GaussianScorer(
                    utils.TimeRolling(obj, period=t_e),
                    threshold=threshold,
                    grace_period=GRACE_PERIOD)

        source = self.get_source(client, in_topics, debug)

        detector = (source
                    .map(self.preprocess, in_topics)
                    .map(self.fit_transform, model)
                    )

        if key_path:
            sender, _ = init_rsa_security(key_path)
            detector = (detector
                        .map(sign_data, sender)
                        .map(encrypt_data, sender)
                        .map(decode_data)
                        )
        detector = self.get_sink(client, in_topics, detector)
        if email is not None:
            email_client = EmailClient(subject="Anomaly detected!", **email)
            detector.sliding_window(2).sink(
                self.send_anomaly_email, email_client)

        try:
            self.run(client, source, detector, debug)
        finally:
            detector.stop()
            print("=== Service stopped ===")
            save_model(recovery_path, in_topics, model)
