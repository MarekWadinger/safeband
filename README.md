# AID: Adaptable and Interpretable Framework for Anomaly Detection

<!-- markdownlint-disable MD013 -->
[![Python application](https://github.com/MarekWadinger/online_outlier_detection/actions/workflows/python-app.yml/badge.svg)](https://github.com/MarekWadinger/online_outlier_detection/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/MarekWadinger/adaptive-interpretable-ad/graph/badge.svg?token=BIS0A7CF1F)](https://codecov.io/gh/MarekWadinger/adaptive-interpretable-ad)
[![Test Status](/reports/test-badge.svg)](https://htmlpreview.github.io/?https://github.com/MarekWadinger/online_outlier_detection/blob/main/reports/junit/report/index.html)
[![Flake8 Status](/reports/flake8-badge.svg)](https://htmlpreview.github.io/?https://github.com/MarekWadinger/online_outlier_detection/blob/main/reports/flake8/report/index.html)
[![DOI](https://zenodo.org/badge/DOI/10.1109/j.eswa.2024.123200.svg)](https://doi.org/10.1016/j.eswa.2024.123200)
<!-- markdownlint-enable MD013 -->

Online outlier detection service for industrial SCADA-based infrastructures for
low-latency detection and change-point adaptation.
The service provides dynamic operating limits based on changing environmental
conditions and sensors aging. This implementation is built upon a robust
foundation, leveraging the power of the open-source libraries
**[river](https://github.com/online-ml/river)**,
**[streamz](https://github.com/python-streamz/streamz)** and
**[human_security](https://github.com/mdipierro/human_security)**, among the
others. Make sure to check out their great work!

### Highlights:

* Interpretable anomaly detector with self-supervised adaptation
* Demonstrates interpretability by providing dynamic operating limits
* Leverages self-learning approach on streamed IoT data
* Utilizes existing SCADA-based industrial infrastruture
* Offers faster response time to incidents due to root cause isolation

![ESwA23 - Graphical Abstract](https://github.com/MarekWadinger/online_outlier_detection/blob/main/publications/ESwA2023/figures/ESwA23%20-%20Graphical%20Abstract.pdf)

![BESS_thresh](https://github.com/MarekWadinger/online_outlier_detection/blob/main/publications/ESwA2023/figures/TERRA_thresh_4days.pdf)

## ⚡️ Quickstart

Get your hand on the algorithm using following Jupyter notebooks and play
around with open-spource example data:

0. [Case Study 0: Outlier Detection on Inverter Temperature](https://github.com/MarekWadinger/online_outlier_detection/blob/main/examples/03_conditional_ae_2023.ipynb)
1. [Case Study 1: Anomaly Detection on BESS Temperature](https://github.com/MarekWadinger/online_outlier_detection/blob/main/examples/03_conditional_ae_2023.ipynb)
2. [Case Study 2: Anomaly Detection on Battery Module Temperature](https://github.com/MarekWadinger/online_outlier_detection/blob/main/examples/04_eco_pack_presov.ipynb)
3. [Comparison Study: One-Class SVM and HalfSpace Trees on SCAB Dataset](https://github.com/MarekWadinger/online_outlier_detection/blob/main/examples/04_eco_pack_presov.ipynb)

## 🏃 Run the services

Our framework is ready to face your challenges with diverse set of suppported
publish-subscribe services:

* [**MQTT**](https://mqtt.org)
* [**Apache Kafka**](https://kafka.apache.org)
* [**Apache Pulsar**](https://pulsar.apache.org)
* Streamed [**DataFrame**](https://pandas.pydata.org)
* TODO: [**NATS**](https://nats.io)

**NOTE**: Messaging can be **signed** and **encrypted** for most of the
services. If you find any related bugs, feel free to
[open an issue](https://github.com/MarekWadinger/online_outlier_detection/issues/new/choose).

### Example Service Usage: MQTT

We demonstrate the usage of the service using
[**MQTT**](https://mqtt.org) protocol. The service is based on
[**paho-mqtt**](https://pypi.org/project/paho-mqtt/) library. The source of data
is a real coffee machine streaming data to MQTT broker.

To start the service, run following line of code in your terminal:

```bash
python rpc_client.py -t "shellies/Shelly3EM-Main-Switchboard-C/emeter/0/power"
```

Note: You can modify the source data stream using attributes:

* `[-f | --config-file]` with path to `config.ini`
(**NOTE**: first valid key value pair is used)
* `[-t | --topic]` to define topic to subscribe to or column in csv file
* `[-k | --key-path]` with path to ssh keys of sender and receiver
(NOTE: if empty, the keys are created)

To start consumer, run following command:

<!-- markdownlint-disable MD013 -->
```bash
python consumer.py -t "shellies/Shelly3EM-Main-Switchboard-C/emeter/0/dynamic_limits"
```
<!-- markdownlint-enable MD013 -->

Note: You can modify the source data stream using attributes:

* `[-f | --config-file]` with path to `config.ini`
(NOTE: first valid key value pair is used)
* `[-t | --topic]` topic of MQTT or column of pd.DataFrame
* `[-k | --key-path]` with path to ssh keys of sender and receiver
(NOTE: if empty, the keys are created)

Query service responds with printed messages as follows:

<!-- markdownlint-disable MD013 -->
```bash
Received message: {"time": "1970-01-01 03:17:11", "anomaly": "0", "level_high":"658.396223558289", "level_low": "635.8731097750442"}
```
<!-- markdownlint-enable MD013 -->

### Example Service Usage: Streamed DataFrame

If you want to stream example dataset use

```bash
python rpc_client.py -t "Average Cell Temperature"
```

where your `config.ini` shall contain

```ini
[file]
path=examples/data/input/average_temperature.csv
output=examples/data/output/dynamic_limits.json
```

Now, let's query the latest limits from data/output/dynamic_limits.json

```bash
python consumer.py -t "Average Cell Temperature"
```

The response is the latest date in `dynamic_limits.json`

```python
{
    "time": datetime.datetime(1970, 1, 1, 14, 52, 42),
    "anomaly": 0,
    "level_high": 1180.92,
    "level_low": 1151.15,
}
```

Note: You can modify the attributes to retrieve thrasholds at any date:

* `[-d | --date]` date as 'Y-m-d H:M:S'

## 🛠 Installation

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## 👐 Contributing

Feel free to contribute in any way you like, we're always open to new ideas and
approaches.

* Feel welcome to
[open an issue](https://github.com/MarekWadinger/online_outlier_detection/issues/new/choose)
if you think you've spotted a bug or a performance issue.

## 💬 Citation

If the service or the algorithm has been useful to you and you would like to
cite it in an scientific publication, please refer to the
[paper](https://doi.org/10.1016/j.eswa.2024.123200)
published in Expert Systems with Applications:

```bibtex
@article{WADINGER2024123200,
  title    = {Adaptable and Interpretable Framework for Anomaly Detection in SCADA-based industrial systems},
  journal  = {Expert Systems with Applications},
  pages    = {123200},
  year     = {2024},
  issn     = {0957-4174},
  doi      = {https://doi.org/10.1016/j.eswa.2024.123200},
  url      = {https://www.sciencedirect.com/science/article/pii/S0957417424000654},
  author   = {Marek Wadinger and Michal Kvasnica},
  keywords = {Anomaly detection, Root cause isolation, Iterative learning, Statistical learning, Self-supervised learning},
}
```

<!--
## 📝 License

This algorithm is free and open-source software licensed under the []().
  -->
