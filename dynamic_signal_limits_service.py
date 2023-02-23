# IMPORTS
import argparse
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


class GaussianScorer(anomaly.base.SupervisedAnomalyDetector):
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
                "scale": self.gaussian.sigma}
        # TODO: consider strict process boundaries
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            real_thresh = norm.ppf((self.threshold/2 + 0.5), **kwargs)
        return real_thresh

                
    def process_one(self, x, t=None):
        if self.gaussian.n_samples == 0:
                self.gaussian._var.mean._mean = x
        
        is_anomaly = self.predict_one(x)
        
        real_thresh = self.limit_one()
        
        if not is_anomaly:
            self = self.learn_one(x, **{"t": t})
        
        return is_anomaly, real_thresh

# FUNCTIONS
def preprocess(
        x,
        col):
    col = [col] if not isinstance(col, list) else col
    if isinstance(x, (pd.Series)):    
        return {"time": x.name.tz_localize(None),
                "data": x[col].to_dict()
        }
    elif isinstance(x, tuple) and isinstance(x[1], (pd.Series)):
        return {"time": x[0].tz_localize(None),
                "data": x[1][col].to_dict()
        }
    elif isinstance(x, dict):
        return  {k: v for k, v in x.items() if k in col} 
    elif isinstance(x, MQTTMessage):
        return {"time": dt.datetime.fromtimestamp(x.timestamp).replace(microsecond=0),
                "data": {x.topic.split("/")[-1]: float(x.payload)}
        }


def fit_transform(
        x, 
        model, 
        model_inv
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


def signal_handler(sig, frame, source, f):
    os.write(sys.stdout.fileno(), b"\nSignal received to stop the app...\n")
    source.stop()
    f.close()
    
    time.sleep(1)
    # Print summary
    d = pd.read_json("data.json", lines=True)
    if not d.empty:
        print_summary(d)
    else:
        print("No data retrieved")
    
    exit(0)
    
    
def process_limits_streaming(
        topic: str,
        data: pd.DataFrame):
    port = 1883
    model = GaussianScorer()
    model_inv = GaussianScorer()
    
    if isinstance(data, str):
        source = Stream.from_mqtt(data, port, topic)
    elif isinstance(data, pd.DataFrame):
        source = Stream.from_iterable(data.iterrows())
    else:
        raise(RuntimeError("Wrong data format."))
    
    detector = source.map(preprocess, topic).map(fit_transform, model, model_inv)
        
    with open("dynamic_limits.json", 'a') as f:
        if isinstance(data, str):
            detector.map(lambda x: json.dumps(x)).to_mqtt(data, port, f"{topic.rsplit('/', 1)[0]}/dynamic_limits", publish_kwargs={"retain":True})
        elif isinstance(data, pd.DataFrame):
            detector.sink(dump_to_file, f)
        source.start()
        
        signal.signal(signal.SIGINT, lambda signalnum, frame: signal_handler(signalnum, frame, source, f))
                      
        while True:
            time.sleep(2)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File to read. If none MQTT is used.", default=None)
    parser.add_argument("-u", "--url", help="MQTT broker URL.", default="mqtt.cloud.uiam.sk")
    #"shellies/Shelly3EM-Main-Switchboard-C/emeter/0/power"
    #"Average Cell Temperature"
    parser.add_argument("-t", "--topic", help="Topic of MQTT or Column of pd.DataFrame", default="shellies/Shelly3EM-Main-Switchboard-C/emeter/0/power")
    args = parser.parse_args()
    
    if args.file and args.url:
        #raise(ValueError("Specify either -f or -u"))
        data = pd.read_csv(args.file, index_col=0)
        data.index = pd.to_datetime(data.index)
    elif args.file:
        data = pd.read_csv(args.file, index_col=0)
        data.index = pd.to_datetime(data.index)
    elif args.url:
        data = args.url
    else:
        raise(ValueError("Specify either -f or -u."))
    topic = args.topic
    
    process_limits_streaming(topic, data)
