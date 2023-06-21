# IMPORTS
from argparse import ArgumentParser, FileType
from configparser import ConfigParser
import datetime as dt
import json
import os
import signal
import sys
import time
import warnings

from paho.mqtt.client import MQTTMessage
import pandas as pd
from river import utils, proba, anomaly
from scipy.stats import norm
from streamz import Stream, Sink

# CONSTANTS
THRESHOLD = 0.99735
GRACE_PERIOD = 60*24
WINDOW = dt.timedelta(hours=24*7)
VAR_SMOOTHING = 1e-9


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


class GaussianScorer(anomaly.base.SupervisedAnomalyDetector):
    """
    Gaussian Scorer for anomaly detection.

    Args:
        threshold (float): Anomaly threshold.
        window_size (int or None): Size of the rolling window.
        period (int or None): Time period for time rolling.
        grace_period (int): Grace period before scoring starts.

    Examples:
    >>> scorer = GaussianScorer(window_size=3, grace_period=2)
    >>> isinstance(scorer, GaussianScorer)
    True
    >>> scorer.gaussian.mu
    0.0
    >>> scorer.learn_one(1).gaussian.mu
    1.0
    >>> scorer.gaussian.sigma
    0.0
    >>> scorer.learn_one(0).gaussian.sigma
    0.7071067811865476
    >>> scorer.limit_one()
    2.625326733368662
    >>> scorer.predict_one(2.625326733368662)
    0
    >>> scorer.score_one(2.625326733368662)
    0.99735

    Anomaly is zero due to grace_period
    >>> scorer.predict_one(2.62532673337)
    0
    >>> scorer.learn_one(1).gaussian.sigma
    0.5773502691896258
    >>> scorer.predict_one(2.62532673337)
    1

    Keeps the sigma due to window_size of 3
    >>> scorer.learn_one(1).gaussian.sigma
    0.5773502691896258
    >>> scorer.process_one(0.5)
    (0, 2.401988677816472)

    Gaussian scorer on time rolling window
    >>> import datetime
    >>> scorer = GaussianScorer()
    >>> scorer.process_one(1, t=datetime.datetime(2022,2,2))
    (0, nan)

    Gaussian scorer without window
    >>> import datetime
    >>> scorer = GaussianScorer(window_size=None, period=None)
    >>> scorer.process_one(1)
    (0, nan)
    """
    def __init__(self,
                 threshold=THRESHOLD,
                 window_size=None,
                 period=WINDOW,
                 grace_period=GRACE_PERIOD):
        self.window_size = window_size
        self.period = period
        if window_size:
            self.gaussian = utils.Rolling(proba.Gaussian(),
                                          window_size=self.window_size)
        elif period:
            self.gaussian = utils.TimeRolling(proba.Gaussian(),
                                              period=self.period)
        else:
            self.gaussian = proba.Gaussian()
        self.grace_period = grace_period
        self.threshold = threshold

    def learn_one(self, x, **kwargs):
        self.gaussian.update(x, **kwargs)
        return self

    def score_one(self, x, t=None):
        if self.gaussian.n_samples < self.grace_period:
            return 0
        return 2 * abs(self.gaussian.cdf(x) - 0.5)

    def predict_one(self, x, t=None):
        score = self.score_one(x)
        if self.gaussian.obj.n_samples > self.grace_period:
            return 1 if score > self.threshold else 0
        else:
            return 0

    def limit_one(self):
        kwargs = {"loc": self.gaussian.mu,
                  "scale":
                      self.gaussian.sigma
                      if not isinstance(self.gaussian.sigma, complex) else 0}
        # TODO: consider strict process boundaries
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            real_thresh = norm.ppf((self.threshold/2 + 0.5), **kwargs)
        return real_thresh

    def process_one(self, x, t=None):
        if self.gaussian.n_samples == 0:
            self.gaussian.obj = self.gaussian._from_state(0, x,
                                                          VAR_SMOOTHING, 1)

        is_anomaly = self.predict_one(x)

        real_thresh = self.limit_one()

        if not is_anomaly:
            if isinstance(self.gaussian, utils.TimeRolling):
                self = self.learn_one(x, **{"t": t})
            else:
                self = self.learn_one(x)

        return is_anomaly, real_thresh


# FUNCTIONS
def preprocess(
        x,
        col):
    """Preprocess the input data.

    Args:
        x (Union[pd.Series, tuple, dict, MQTTMessage, bytes]): The input data
        to be preprocessed.
        col (Union[str, List[str]]): The column(s) to be extracted from the
        input data.

    Returns:
        dict: The preprocessed data.

    Examples:
        >>> series = pd.Series([1.], name=pd.to_datetime('2023-01-01'),
        ...                    index=["sensor_1"])
        >>> preprocess(series, 'sensor_1')
        {'time': Timestamp('2023-01-01 00:00:00'), 'data': {'sensor_1': 1.0}}

        >>> series_tuple = (pd.to_datetime('2023-01-01'), series)
        >>> preprocess(series_tuple, 'sensor_1')
        {'time': Timestamp('2023-01-01 00:00:00'), 'data': {'sensor_1': 1.0}}

        >>> data_dict = {'time': pd.to_datetime('2023-01-01'), 'sensor_1': 1.}
        >>> out = preprocess(data_dict, ['sensor_1'])
        >>> out.keys(), out['data'].keys()
        (dict_keys(['time', 'data']), dict_keys(['sensor_1']))

        >>> mqtt_message = MQTTMessage()
        >>> mqtt_message.timestamp = 1672527600.0
        >>> mqtt_message.payload = b'1.'
        >>> mqtt_message.topic = b'sensors/sensor_1'
        >>> out = preprocess(mqtt_message, '1')
        >>> out.keys(), out['data'].keys()
        (dict_keys(['time', 'data']), dict_keys(['sensor_1']))

        >>> binary_data = b'1.0'
        >>> out = preprocess(binary_data, 'sensor_1')
        >>> out.keys(), out['data'].keys()
        (dict_keys(['time', 'data']), dict_keys(['sensor_1']))
    """
    if isinstance(x, pd.Series):
        col = [col] if not isinstance(col, list) else col
        return {"time": x.name.tz_localize(None),
                "data": x[col].to_dict()
                }
    elif isinstance(x, tuple) and isinstance(x[1], (pd.Series)):
        col = [col] if not isinstance(col, list) else col
        return {"time": x[0].tz_localize(None),
                "data": x[1][col].to_dict()
                }
    elif isinstance(x, dict):
        return {"time": dt.datetime.now().replace(microsecond=0),
                "data": {k: v for k, v in x.items() if k in col}
                }
    elif isinstance(x, MQTTMessage):
        return {"time": dt.datetime
                .fromtimestamp(x.timestamp).replace(microsecond=0),
                "data": {x.topic.split("/")[-1]: float(x.payload)}
                }
    elif isinstance(x, bytes):
        return {"time": dt.datetime.now().replace(microsecond=0),
                "data": {col: float(x.decode("utf-8"))}
                }


def fit_transform(
        x,
        model,
        model_inv
):
    """Apply anomaly detection model to the input data.

    The function applies the provided anomaly detection model to the input
    data and returns the result as a dictionary.

    Args:
        x (dict): The input data dictionary.
        model: The anomaly detection model.
        model_inv: The inverse anomaly detection model.

    Returns:
        dict: The processed data dictionary.

    Examples:
        >>> x = {"time": dt.datetime(2022,1,1),
        ...      "data": {"feature1": 0.5, "feature2": 1.2, "feature3": -0.8}}
        >>> model = GaussianScorer()
        >>> model_inv = GaussianScorer()
        >>> result = fit_transform(x, model, model_inv)
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
    is_anomaly, real_thresh = model.process_one(x_, x["time"])
    _, real_thresh_ = model_inv.process_one(-x_, x["time"])
    return {"time": str(x["time"]),
            # **x["data"], # Comment out to lessen the size of payload
            "anomaly": is_anomaly,
            "level_high": real_thresh,
            "level_low": -real_thresh_
            }


def dump_to_file(x, f):  # pragma: no cover
    print(json.dumps(x), file=f)


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
    if config.get("path"):
        d = pd.read_json("data/output/dynamic_limits.json", lines=True)
        if not d.empty:
            print_summary(d)
        else:
            print("No data retrieved")
    # TODO: Find out how to flush kafka
    # if config.get("bootstrap.servers"):
    #    detector.flush()

    exit(0)


def get_source(
        config: dict,
        topic: str,
        debug: bool = False):
    """Get the data source based on the provided configuration.

    The function returns a data source stream object based on the
    configuration settings.
    If the 'path' key is present in the config, it returns a stream from an
    iterable of
    rows in the 'data' dictionary. If the 'host' key is present, it returns a
    stream from
    MQTT messages with the specified topic. If the 'bootstrap.servers' key is
    present,
    it returns a stream from Kafka messages with the specified topic. If none
    of the expected keys are found, it raises a RuntimeError.

    Args:
        config (dict): The configuration dictionary.
        topic (str): The topic to subscribe to for MQTT or Kafka sources.
        debug (bool, optional): Enable debug mode. Defaults to False.

    Returns:
        stream.Stream: The data source stream object.

    Raises:
        RuntimeError: If the data format is incorrect.

    Examples:
    >>> config = {
    ...     "path": "path/to/input/data.csv",
    ...     "data": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})}
    >>> topic = "test"
    >>> source = get_source(config, topic)
    >>> type(source)
    <class 'streamz.sources.from_iterable'>

    >>> source = get_source(config, topic, debug=True)
    >>> type(source)
    <class 'streamz.core.Stream'>

    >>> config = {"host": "mqtt.server", "port": 1883}
    >>> topic = "test"
    >>> source = get_source(config, topic)
    >>> type(source)
    <class 'streamz.sources.from_mqtt'>

    >>> config = {"bootstrap.servers": "kafka.server:9092",
    ...           "group.id": "consumer-group"}
    >>> topic = "kafka-topic"
    >>> source = get_source(config, topic)
    >>> type(source)
    <class 'streamz.sources.from_kafka'>

    >>> config = {"invalid": "config"}
    >>> topic = "test"
    >>> source = get_source(config, topic)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    RuntimeError: Wrong data format.
    """
    if config.get("path"):
        if debug:
            source = Stream()
        else:
            source = Stream.from_iterable(config['data'].iterrows())
    elif config.get("host"):
        source = Stream.from_mqtt(**config, topic=topic)
    elif config.get("bootstrap.servers"):
        source = Stream.from_kafka(
            [topic], {**config, 'group.id': 'detection_service'})
    else:
        raise (RuntimeError("Wrong data format."))
    return source


def process_limits_streaming(
        config: dict,
        topic: str,
        debug: bool = False):
    """Process the limits in a streaming manner.

    The function sets up the necessary components for streaming processing of
    limits.
    It creates instances of the GaussianScorer model for anomaly detection,
    prepares
    the data source based on the configuration, and performs the required
    transformations.
    The processed data is then stored or published based on the configuration.

    Args:
        config (dict): The configuration dictionary.
        topic (str): The topic to subscribe to for MQTT or Kafka sources.
        debug (bool, optional): Enable debug mode. Defaults to False.

    Examples:
    >>> config = {"path": "data/test.csv"}
    >>> topic = "A"
    >>> process_limits_streaming(config, topic, debug=True)
    === Debugging started... ===
    === Debugging finished with success... ===
    """
    model = GaussianScorer()
    model_inv = GaussianScorer()

    if config.get("path"):
        data = pd.read_csv(config['path'], index_col=0)
        data.index = pd.to_datetime(data.index, utc=True)
        config['data'] = data

    source = get_source(config, topic, debug)

    detector = source.map(preprocess, topic).map(
        fit_transform, model, model_inv)

    with open("data/output/dynamic_limits.json", 'a') as f:
        if config.get("path"):
            detector.sink(dump_to_file, f)
        elif config.get("host"):  # pragma: no cover
            topic = f"{topic.rsplit('/', 1)[0]}/dynamic_limits"
            detector.map(lambda x: json.dumps(x)).to_mqtt(
                **config, topic=topic, publish_kwargs={"retain": True})
        elif config.get("bootstrap.servers"):  # pragma: no cover
            topic = "dynamic_limits"
            detector.map(lambda x: (str(x), "dynamic_limits")
                         ).to_kafka(topic, config)

        if debug:
            print("=== Debugging started... ===")
            for row in data.iterrows():
                source.emit(row)
            print("=== Debugging finished with success... ===")
        else:  # pragma: no cover
            source.start()

            signal.signal(signal.SIGINT, lambda signalnum,
                          frame: signal_handler(
                              signalnum, frame, detector, config))

            while True:
                time.sleep(2)


def get_config(config_parser):  # pragma: no cover
    if (config_parser.has_option('file', 'path') and
            config_parser.get('file', 'path')):
        config = dict(config_parser['file'])
    elif (config_parser.has_section('mqtt') and
          config_parser.has_option('mqtt', 'host') and
          config_parser.has_option('mqtt', 'port') and
          config_parser.get('mqtt', 'host') and
          config_parser.get('mqtt', 'port')):
        config = dict(config_parser['mqtt'])
        config['port'] = int(config['port'])
    elif (config_parser.has_section('kafka') and
          config_parser.has_option('kafka', 'bootstrap.servers') and
          config_parser.get('kafka', 'bootstrap.servers')):
        config = dict(config_parser['kafka'])
    else:
        raise ValueError("Missing configuration.")
    return config


if __name__ == '__main__':  # pragma: no cover
    parser = ArgumentParser()
    parser.add_argument('-f', '--config_file',
                        type=FileType('r'), default='config.ini')
    # "shellies/Shelly3EM-Main-Switchboard-C/emeter/0/power"
    # "Average Cell Temperature"
    parser.add_argument("-t", "--topic",
                        help="Topic of MQTT or Column of pd.DataFrame",
                        default="Average Cell Temperature")
    parser.add_argument("-d", "--debug",
                        help="Debug the file using loop as source",
                        default=False, type=bool)
    args = parser.parse_args()

    config_parser = ConfigParser()
    config_parser.read_file(args.config_file)

    # TODO: Handle possible errorous scenarios
    config = get_config(config_parser)

    process_limits_streaming(config, args.topic, args.debug)
