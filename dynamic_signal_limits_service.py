# IMPORTS
from argparse import ArgumentParser, FileType
from configparser import ConfigParser
import datetime as dt
import json
import os
import signal
import sys
import time
import typing
import warnings

from paho.mqtt.client import MQTTMessage
import pandas as pd
from river import utils, proba, anomaly
from scipy.stats import norm
from streamz import Stream, Sink

from functions.anomaly import GaussianScorer

# CONSTANTS
GRACE_PERIOD=60*24
WINDOW = dt.timedelta(hours=24*7)


# DEFINITIONS
@Stream.register_api()
class to_mqtt(Sink):
    """
    Send data to MQTT broker

    See also ``sources.from_mqtt``.

    Requires ``paho.mqtt``

    :param host: str
    :param port: int
    :param topic: str
    :param keepalive: int
        See mqtt docs - to keep the channel alive
    :param client_kwargs:
        Passed to the client's ``connect()`` method
    """
    def __init__(self, upstream, host, port, topic, keepalive=60, client_kwargs=None, publish_kwargs=None,
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
            self.client.connect(self.host, self.port, self.keepalive, **self.c_kw)
        # TODO: wait on successful delivery
        self.client.publish(self.topic, x, **self.p_kw)

    def destroy(self):
        self.client.disconnect()
        self.client = None
        super().destroy()


# FUNCTIONS
def preprocess(
        x,
        col):
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
        return  {k: v for k, v in x.items() if k in col} 
    elif isinstance(x, MQTTMessage):
        return {"time": dt.datetime.fromtimestamp(x.timestamp).replace(microsecond=0),
                "data": {x.topic.split("/")[-1]: float(x.payload)}
        }
    elif isinstance(x, bytes):
        return {"time": dt.datetime.now().replace(microsecond=0),
                "data": {col: float(x.decode("utf-8"))}
        }


def fit_transform(
        x, 
        model: GaussianScorer, 
        model_inv: GaussianScorer
        ):
    # TODO: replace x_ for multidimensional implementation
    x_ = next(iter(x["data"].values()))
    is_anomaly, real_thresh = model.process_one(x_, x["time"])
    _, real_thresh_ = model_inv.process_one(-x_, x["time"])
    return {"time": str(x["time"]),
            #**x["data"], # Comment out to lessen the size of payload
            "anomaly":is_anomaly,
            "level_high":real_thresh, 
            "level_low":-real_thresh_
            }


def dump_to_file(x, f):
    print(json.dumps(x), file=f)


def print_summary(df):
    text = (
            f"Proportion of anomalous samples: "
            f"{sum(df['anomaly'])/len(df['anomaly'])*100:.02f}%\n"
            f"Total number of anomalous events: "
            f"{sum(pd.Series(df['anomaly']).diff().dropna() == 1)}")
    print(text) 


def signal_handler(sig, frame, detector, config):
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
    #if config.get("bootstrap.servers"):
    #    detector.flush()
    
    exit(0)
    
    
def process_limits_streaming(
        config: dict,
        topic: str,
        debug: bool = False):
    model = GaussianScorer(
        utils.TimeRolling(proba.Gaussian(), period=WINDOW),
        grace_period=GRACE_PERIOD)
    model_inv = GaussianScorer(
        utils.TimeRolling(proba.Gaussian(), period=WINDOW),
        grace_period=GRACE_PERIOD)
    
    if config.get("path"):
        data = pd.read_csv(config['path'], index_col=0)
        data.index = pd.to_datetime(data.index, utc=True)
        if debug:
            source = Stream()
        else:
            source = Stream.from_iterable(data.iterrows())
    elif config.get("host"):
        source = Stream.from_mqtt(**config, topic=topic)
    elif config.get("bootstrap.servers"):
        source = Stream.from_kafka([topic], {**config, 'group.id': 'detection_service'})
    else:
        raise(RuntimeError("Wrong data format."))
    
    detector = source.map(preprocess, topic).map(fit_transform, model, model_inv)

    with open("data/output/dynamic_limits.json", 'a') as f:
        if config.get("path"):
            detector.sink(dump_to_file, f)
        elif config.get("host"):
            topic = f"{topic.rsplit('/', 1)[0]}/dynamic_limits"
            detector.map(lambda x: json.dumps(x)).to_mqtt(**config, topic=topic, publish_kwargs={"retain":True})
        elif config.get("bootstrap.servers"):
            topic = "dynamic_limits"
            detector.map(lambda x: (str(x), "dynamic_limits")).to_kafka(topic, config)

        if debug:
            print("=== Debugging started... ===")
            for row in data.iterrows():
                source.emit(row)
            print("=== Debugging finished with success... ===")
        else:
            source.start()
            
            signal.signal(signal.SIGINT, lambda signalnum, frame: signal_handler(signalnum, frame, detector, config))
                        
            while True:
                time.sleep(2)

    
def get_config(config_parser):
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--config_file', type=FileType('r'), default='config.ini')
    #"shellies/Shelly3EM-Main-Switchboard-C/emeter/0/power"
    #"Average Cell Temperature"
    parser.add_argument("-t", "--topic", help="Topic of MQTT or Column of pd.DataFrame", default="Average Cell Temperature")
    parser.add_argument("-d", "--debug", help="Debug the file using loop as source", default=False, type=bool)
    args = parser.parse_args()
    
    config_parser = ConfigParser()
    config_parser.read_file(args.config_file)
    
    # TODO: Handle possible errorous scenarios
    config = get_config(config_parser)

    process_limits_streaming(config, args.topic, args.debug)
