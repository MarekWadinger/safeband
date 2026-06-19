<p align="center">
  <img src="docs/assets/logo.png" alt="safeband" width="200">
</p>

<h1 align="center">safeband</h1>

<p align="center">
  <strong>Stop babysitting static thresholds that go stale — get explainable operating limits that adapt themselves.</strong>
</p>

<p align="center">
  <em>Online, interpretable anomaly detection for industrial sensor streams.<br/>Learns self-adjusting operating limits that track drift and sensor aging, isolates the root-cause signal, and explains every alarm — no labels, no retraining.</em>
</p>

<!-- markdownlint-disable MD013 -->
<p align="center">
  <a href="https://github.com/MarekWadinger/safeband/actions/workflows/python-app.yml"><img src="https://github.com/MarekWadinger/safeband/actions/workflows/python-app.yml/badge.svg" alt="Python application"></a>
  <a href="https://codecov.io/gh/MarekWadinger/safeband"><img src="https://codecov.io/gh/MarekWadinger/safeband/graph/badge.svg" alt="codecov"></a>
  <a href="https://doi.org/10.1016/j.eswa.2024.123200"><img src="https://zenodo.org/badge/DOI/10.1016/j.eswa.2024.123200.svg" alt="DOI"></a>
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+">
</p>
<!-- markdownlint-enable MD013 -->

---

## Highlights

- **Interpretable by construction** — every alarm names the dynamic operating limit it crossed, not an opaque score.
- **Self-adapting** — limits track environmental change and sensor aging online; no manual re-tuning, no batch retraining.
- **Change-point aware** — detects regime shifts and re-bases the safe range instead of drowning in false alarms.
- **Root-cause isolation** — built-in fault diagnosis points to the signal that drove an event.
- **Streaming-native** — built on [`river`](https://github.com/online-ml/river) + [`streamz`](https://github.com/python-streamz/streamz): single-pass, constant-memory.
- **SCADA-ready** — drops onto existing industrial infrastructure (MQTT / NATS / Pulsar ingest), secured with [`human_security`](https://github.com/mdipierro/human_security).

## Overview

Online outlier detection for industrial SCADA-based infrastructures: low-latency detection with change-point adaptation. safeband provides dynamic operating limits that adapt to changing environmental conditions and sensor aging.

[Image: Graphical Abstract](https://github.com/MarekWadinger/safeband/blob/main/publications/ESwA2023/figures/ESwA23%20-%20Graphical%20Abstract%20-%20Latex%20font.pdf)

[Image: Dynamic operating limits and detection on utility-scale battery temperature profile](https://github.com/MarekWadinger/safeband/blob/main/publications/ESwA2023/figures/TERRA_thresh_4days.pdf)

## ⚡️ Quickstart

Get your hand on the algorithm using following Jupyter notebooks and play
around with open-source example data:

0. [Case Study 0: Outlier Detection on Inverter Temperature](https://github.com/MarekWadinger/safeband/blob/main/examples/01_univariate_pc_2023.ipynb)
1. [Case Study 1: Anomaly Detection on BESS Temperature](https://github.com/MarekWadinger/safeband/blob/main/examples/03_conditional_ae_2023.ipynb)
2. [Case Study 2: Anomaly Detection on Battery Module Temperature](https://github.com/MarekWadinger/safeband/blob/main/examples/04_eco_pack_presov.ipynb)
3. [Comparison Study: One-Class SVM and HalfSpace Trees on SKAB Dataset](https://github.com/MarekWadinger/safeband/blob/main/examples/comparison.ipynb)
4. [Benchmark Study: AID on the TSB-AD benchmark (VUS-PR protocol)](https://github.com/MarekWadinger/safeband/blob/main/examples/06_tsb_ad_benchmark.ipynb)

## 🏃 Run the services

Our framework is ready to face your challenges with diverse set of supported
publish-subscribe services:

* [**MQTT**](https://mqtt.org)
* [**Apache Kafka**](https://kafka.apache.org)
* [**Apache Pulsar**](https://pulsar.apache.org)
* [**NATS**](https://nats.io)
* Streamed [**DataFrame**](https://pandas.pydata.org)

[**NATS**](https://nats.io) is a first-class transport. Add a `[nats]`
section pointing at your server(s) and run the service just like the
MQTT example below:

<!-- markdownlint-disable MD013 -->
```ini
[nats]
servers=nats://localhost:4222   ; comma-separate the value for a cluster
```

```bash
uv run python rpc_client.py -f example.ini -t "plant/temperature"
uv run python consumer.py  -f example.ini -t "plant/temperature"
```
<!-- markdownlint-enable MD013 -->

The detector subscribes to each input subject (`-t` / `in_topics`) and
publishes results to derived subjects: `<topic>anomaly` for the flag and
`<topic>_DOL_high` / `<topic>_DOL_low` for the dynamic limits in the
univariate case, or per-signal `<signal>_DOL_high` / `<signal>_DOL_low` /
`<signal>_root_cause` in the multivariate case. As with the other
transports, messages can be signed and encrypted.

**NOTE**: Messaging can be **signed** and **encrypted** for most of the
services. If you find any related bugs, feel free to
[open an issue](https://github.com/MarekWadinger/safeband/issues/new/choose).

### Example Service Usage: MQTT

We demonstrate the usage of the service using
[**MQTT**](https://mqtt.org) protocol. The service is based on
[**paho-mqtt**](https://pypi.org/project/paho-mqtt/) library. The source of data
is a real coffee machine streaming data to MQTT broker.

The MQTT example requires a config file with an `[mqtt]` section (e.g. the
provided `example.ini`), while the default `config.ini` is set up for the
Streamed-DataFrame example.

To start the service, run following line of code in your terminal:

<!-- markdownlint-disable MD013 -->
```bash
uv run python rpc_client.py -f example.ini -t "shellies/Shelly3EM-Main-Switchboard-C/emeter/0/power"
```
<!-- markdownlint-enable MD013 -->

Note: You can modify the source data stream using attributes:

* `[-f | --config-file]` with path to `config.ini`
(**NOTE**: first valid key value pair is used)
* `[-t | --in-topics]` to define topic to subscribe to or column in csv file
* `[-k | --key-path]` with path to ssh keys of sender and receiver
(NOTE: if empty, the keys are created)

To start consumer, run following command:

<!-- markdownlint-disable MD013 -->
```bash
uv run python consumer.py -f example.ini -t "shellies/Shelly3EM-Main-Switchboard-C/emeter/0/dynamic_limits"
```
<!-- markdownlint-enable MD013 -->

Note: You can modify the source data stream using attributes:

* `[-f | --config-file]` with path to `config.ini`
(NOTE: first valid key value pair is used)
* `[-t | --in-topics]` topic of MQTT or column of pd.DataFrame
* `[-k | --key-path]` with path to ssh keys of sender and receiver
(NOTE: if empty, the keys are created)

Query service responds with printed messages as follows:

<!-- markdownlint-disable MD013 -->
```bash
Received message: {"time": "1970-01-01 03:17:11", "anomaly": 0, "level_high": 658.396223558289, "level_low": 635.8731097750442}
```
<!-- markdownlint-enable MD013 -->

### Example Service Usage: Streamed DataFrame

If you want to stream example dataset use

```bash
uv run python rpc_client.py -t "Average Cell Temperature"
```

where your `config.ini` shall contain

```ini
[file]
path=examples/data/input/average_temperature.csv
output=examples/data/output/dynamic_limits.json
```

Now, let's query the latest limits from examples/data/output/dynamic_limits.json

```bash
uv run python consumer.py -t "Average Cell Temperature"
```

The response is the entry of `dynamic_limits.json` closest to the current
date

```python
{
    "time": datetime.datetime(1970, 1, 1, 14, 52, 42),
    "anomaly": 0,
    "level_high": 1180.92,
    "level_low": 1151.15,
}
```

## 🛠 Installation

To install the necessary dependencies, we recommend using [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Alternatively, you can use Docker to set up the environment. Follow these steps:

1. Build the Docker image:

    ```bash
    docker build -t aid .
    ```

2. Run the Docker container:

    ```bash
    docker run -it --rm aid
    ```

You can also use Docker Compose to manage the local services. Simply start the services with:

  ```bash
  docker-compose up
  ```

## 👐 Contributing

Feel free to contribute in any way you like, we're always open to new ideas and
approaches.

* Feel welcome to
[open an issue](https://github.com/MarekWadinger/safeband/issues/new/choose)
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
