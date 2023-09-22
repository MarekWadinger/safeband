# Adaptable and Interpretable Framework for Anomaly Detection

<!-- markdownlint-disable MD013 -->
[![Python application](https://github.com/MarekWadinger/online_outlier_detection/actions/workflows/python-app.yml/badge.svg)](https://github.com/MarekWadinger/online_outlier_detection/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/MarekWadinger/online_outlier_detection/branch/main/graph/badge.svg?token=BIS0A7CF1F)](https://codecov.io/gh/MarekWadinger/online_outlier_detection)
[![Test Status](/reports/test-badge.svg)](https://htmlpreview.github.io/?https://github.com/MarekWadinger/online_outlier_detection/blob/main/reports/junit/report/index.html)
[![Flake8 Status](/reports/flake8-badge.svg)](https://htmlpreview.github.io/?https://github.com/MarekWadinger/online_outlier_detection/blob/main/reports/flake8/report/index.html)
[![DOI](https://zenodo.org/badge/DOI/10.1109/PC58330.2023.10217717.svg)](https://doi.org/10.1109/PC58330.2023.10217717)
<!-- markdownlint-enable MD013 -->

Online outlier detection service for existing real-time infrastructures for
low-latency detection and change-point adaptation.
The service provides dynamic process limits based on changing environmental
conditions and sensors aging. This implementation is built upon a robust
foundation, leveraging the power of the open-source libraries
**[river](https://github.com/online-ml/river)**
and **[streamz](https://github.com/python-streamz/streamz)** and
**[human_security](https://github.com/mdipierro/human_security)**, among the
others. Make sure to check out their great work!

The main benefits of the  solution are that it:

* Enriches interpretable anomaly detection with adaptive capabilities
* Isolates root cause of anomalies while considering interactions
* Uses self-learning approach on streamed IoT data
* Demonstrates interpretability by providing process limits for signals
* Provides comparable detection accuracy to established general methods

![ESwA23 - Graphical Abstract](https://github.com/MarekWadinger/online_outlier_detection/assets/50716630/68049357-9fdf-43db-8144-ef86403606ef)

![BESS_thresh](https://github.com/MarekWadinger/online_outlier_detection/assets/50716630/6c2da80b-ee2b-46f0-8aa1-3bdace9f3229)

## ⚡️ Quickstart

Get your hand on the algorithm using this
[notebook](https://github.com/MarekWadinger/online_outlier_detection/blob/main/examples/01_univariate_pc_2023.ipynb)
and play around with example data.

## 🏃 Run the services

### Stream MQTT messages

To start the service, run following line of code in your terminal:

```bash
python client.py -t "shellies/Shelly3EM-Main-Switchboard-C/emeter/0/power"
```

Note: You can modify the source data stream using attributes:

* `[-f | --config-file]` with path to `config.ini`
(NOTE: first valid key value pair is used)
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

### Stream file messages

If you want to stream example dataset use

```bash
python client.py -t "Average Cell Temperature"
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
{'time': datetime.datetime(1970, 1, 1, 14, 52, 42),
 'anomaly': 0,
 'level_high': 1180.92,
 'level_low': 1151.15}
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
[paper](https://doi.org/10.5281/zenodo.8128206)
in Proceedings of the 2023 24th International Conference on Process Control,
IEEE:

```bibtex
@inproceedings{Wadinger2023,
  author    = {Wadinger, Marek and Kvasnica, Michal},
  booktitle = {2023 24th International Conference on Process Control (PC)},
  title     = {Real-Time Outlier Detection with Dynamic Process Limits},
  year      = {2023},
  doi       = {10.1109/PC58330.2023.10217717},
  month     = {June}
}
```

<!--
## 📝 License

This algorithm is free and open-source software licensed under the []().
  -->
