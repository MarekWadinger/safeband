# IMPORTS
import datetime as dt
import json
import os
import signal
import sys
import time

import pandas as pd
from paho.mqtt.client import MQTTMessage
from river import proba, utils
from streamz import Sink, Stream

from functions.anomaly import GaussianScorer
from functions.encryption import (decode_data, encrypt_data, init_rsa_security,
                                  sign_data)
from functions.safe_streamz import map  # noqa: E402, F401
from functions.utils import common_prefix

# CONSTANTS
GRACE_PERIOD = 60*24
WINDOW = dt.timedelta(hours=24*7)

open_files = []


# DEFINITIONS
@Stream.register_api()
class to_mqtt(Sink):
    """
    Initialize the to_mqtt instance.

    Args:
        upstream (Stream): Upstream stream.
        host (str): MQTT broker host.
        port (int): MQTT broker port.
        topic (str): MQTT topic.
        keepalive (int): Keepalive duration.
        client_kwargs (dict): Additional arguments for MQTT client connect.
        publish_kwargs (dict): Additional arguments for MQTT publish.
        **kwargs: Additional keyword arguments.

    Examples:
    >>> out_msg = bytes(str(dt.datetime.now()), encoding='utf-8')
    >>> mqtt_sink = to_mqtt(
    ...     Stream(), host="mqtt.eclipseprojects.io",
    ...     port=1883, topic='test', publish_kwargs={"retain":True})
    >>> mqtt_sink.update(out_msg)

    Check the message
    >>> import paho.mqtt.subscribe as subscribe
    >>> msg = subscribe.simple(hostname="mqtt.eclipseprojects.io",
    ...                        topics="test")
    >>> msg.payload == out_msg
    True
    """
    def __init__(self, upstream, host, port, topic, keepalive=60,
                 client_kwargs=None, publish_kwargs=None,
                 **kwargs):
        self.host = host
        self.port = port
        self.c_kw = client_kwargs or {}
        self.p_kw = publish_kwargs or {}
        self.client = None
        self.topic = topic
        self.keepalive = keepalive
        super().__init__(upstream, ensure_io_loop=True, **kwargs)

    def update(self, x, who=None, metadata=None):
        import paho.mqtt.client as mqtt
        if self.client is None:
            self.client = mqtt.Client(clean_session=True)
            self.client.connect(self.host, self.port, self.keepalive,
                                **self.c_kw)
        # TODO: wait on successful delivery
        self.client.publish(self.topic, x, **self.p_kw)

    def destroy(self):  # pragma: no cover
        self.client.disconnect()
        self.client = None
        super().destroy()


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


def signal_handler(sig, frame, detector, config):  # pragma: no cover
    os.write(sys.stdout.fileno(), b"\nSignal received to stop the app...\n")
    detector.stop()

    time.sleep(1)
    # Print summary
    if config.get("output"):
        for file in open_files:
            file.close()
            print(f"Output saved to {file.name}")
        try:
            d = pd.read_json(config.get("output"), lines=True)
            if not d.empty:
                print_summary(d)
            else:
                print("No data retrieved")
        except Exception:
            print("Cannot show summary possibly due to encryption.")
    # TODO: Find out how to flush kafka
    # if config.get("bootstrap.servers"):
    #     detector.flush()

    exit(0)


class RpcOutlierDetector(object):
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
            return {"time": x.name.tz_localize(None),
                    "data": x[topics].to_dict()
                    }
        elif isinstance(x, tuple) and isinstance(x[1], (pd.Series)):
            return {"time": x[0].tz_localize(None),
                    "data": x[1][topics].to_dict()
                    }
        elif isinstance(x, dict):
            return {"time": dt.datetime.now().replace(microsecond=0),
                    "data": {k: v for k, v in x.items() if k in topics}
                    }
        elif isinstance(x, MQTTMessage):
            return {"time": dt.datetime
                    .fromtimestamp(x.timestamp).replace(microsecond=0),
                    "data": {x.topic.split("/")[-1]: float(x.payload)}
                    }
        elif isinstance(x, bytes):
            return {"time": dt.datetime.now().replace(microsecond=0),
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
        # TODO: replace x_ for multidimensional implementation
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

    def get_source(
            self,
            config: dict,
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
        ...     "path": "path/to/input/data.csv",
        ...     "data": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})}
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
        <class 'streamz.sources.from_mqtt'>

        >>> config = {"bootstrap.servers": "kafka.server:9092",
        ...           "group.id": "consumer-group"}
        >>> topics = ["kafka-topics"]
        >>> source = obj.get_source(config, topics)
        >>> type(source)
        <class 'streamz.sources.from_kafka'>

        >>> config = {"service_url": "pulsar://localhost:6650"}
        >>> topics = ["pulsar-topics"]
        >>> source = obj.get_source(config, topics)
        >>> type(source)
        <class 'streamz_pulsar.sources.from_pulsar.from_pulsar'>

        >>> config = {"invalid": "config"}
        >>> topics = ["test"]
        >>> source = obj.get_source(config, topics)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        RuntimeError: Wrong data format.
        """  # noqa: E501
        if config.get("path"):
            if debug:
                source = Stream()
            else:
                source = Stream.from_iterable(config['data'].iterrows())
        elif config.get("host"):
            source = Stream.from_mqtt(
                **config, topic=[(topic, 0) for topic in topics])
        elif config.get("bootstrap.servers"):
            source = Stream.from_kafka(
                topics, {**config, 'group.id': 'detection_service'})
        elif config.get("service_url"):
            source = Stream.from_pulsar(
                config.get("service_url"),
                topics,
                subscription_name='detection_service')
        else:
            raise (RuntimeError("Wrong data format."))
        return source

    def get_sink(
            self,
            config: dict,
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
        if config.get("path") and config.get("output"):
            f = open(config.get("output"), 'a')
            open_files.append(f)
            detector.sink(self.dump_to_file, f)
        elif config.get("host"):  # pragma: no cover
            detector.map(lambda x: json.dumps(x)).to_mqtt(
                **config, topic=topic, publish_kwargs={"retain": True})
        # TODO: add coverage test
        elif config.get("bootstrap.servers"):  # pragma: no cover
            detector.map(lambda x: (str(x), "dynamic_limits")
                         ).to_kafka(topic, config)
        elif config.get("service_url"):  # pragma: no cover
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

    def start(
            self,
            config: dict,
            topics: list,
            key_path: str,
            debug: bool = False):
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
        >>> config = {"path": "tests/test.csv", "output": "tests/output.json"}
        >>> topics = ["A"]
        >>> obj = RpcOutlierDetector()
        >>> obj.start(config, topics, key_path=".temp", debug=True)
        === Debugging started... ===
        === Debugging finished with success... ===
        """
        # TODO: Move to encryption.py
        sender, _ = init_rsa_security(key_path)

        model = GaussianScorer(
            utils.TimeRolling(proba.Gaussian(), period=WINDOW),
            grace_period=GRACE_PERIOD)

        if config.get("path"):
            data = pd.read_csv(config['path'], index_col=0)
            data.index = pd.to_datetime(data.index, utc=True)
            config['data'] = data

        source = self.get_source(config, topics, debug)

        detector = (source
                    .map(self.preprocess, topics)
                    .map(self.fit_transform, model)
                    .map(sign_data, sender)
                    .map(encrypt_data, sender)
                    .map(decode_data)
                    )

        detector = self.get_sink(config, topics, detector)

        # TODO: handle combination of debug and remote broker
        if debug and config.get("path"):
            print("=== Debugging started... ===")
            for row in data.head().iterrows():
                source.emit(row)
            for file in open_files:
                file.close()
            print("=== Debugging finished with success... ===")
        else:  # pragma: no cover
            source.start()

            signal.signal(
                signal.SIGINT, lambda signalnum,
                frame: signal_handler(
                    signalnum, frame, detector, config)
                )

            while True:
                if source.stopped:
                    break
                time.sleep(2)
