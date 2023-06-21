# Real-Time Outlier Detection with Dynamic Process Limits

[![Python application](https://github.com/MarekWadinger/online_outlier_detection/actions/workflows/python-app.yml/badge.svg)](https://github.com/MarekWadinger/online_outlier_detection/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/MarekWadinger/online_outlier_detection/branch/main/graph/badge.svg?token=BIS0A7CF1F)](https://codecov.io/gh/MarekWadinger/online_outlier_detection)
[![Test Status](/reports/test-badge.svg)](https://htmlpreview.github.io/?https://github.com/MarekWadinger/online_outlier_detection/blob/main/reports/junit/report/index.html)
[![Flake8 Status](/reports/flake8-badge.svg)](https://htmlpreview.github.io/?https://github.com/MarekWadinger/online_outlier_detection/blob/main/reports/flake8/report/index.html)
[![DOI](https://img.shields.io/badge/DOI-arXiv.2301.13527-brightgreen)](https://doi.org/10.48550/arXiv.2301.13527)

Online outlier detection service for existing real-time infrastructures for low-latency detection and change-point adaptation.
The service provides dynamic process limits based on changing environmental conditions and sensors aging.

The main benefits of the  solution are that it:

* Keeps existing IT infrastructure, saving costs, and does
not require operator retraining
* Automates alerting thresholds setup for a high number of
signals
* Automates alerting for signals with no a priori knowledge
of process limits
* Assesses changing environmental conditions and device
aging
* Uses self-learning approach on streamed data

![Average_Cell_Temperature_sliding_thresh](https://github.com/MarekWadinger/online_outlier_detection/assets/50716630/427586d8-9858-4cf2-9aaa-1ee5407416bd)

## ⚡️ Quickstart

Get your hand on the algorithm using this [notebook](https://github.com/MarekWadinger/online_outlier_detection/blob/main/online_outlier_detection.ipynb) to play around with example data.

As a quick example, we will run the online service in the terminal and use query the latest dynamic limits for our process.

To start the service, run following line of code in your terminal:

```bash
python3 dynamic_signal_limits_service.py -f config.ini -t "shellies/Shelly3EM-Main-Switchboard-C/emeter/0/power"
```

Note: You can modify the source data stream using attributes:

* `[-f | --config_file]` with path to `config.ini` (NOTE: first defined key value pair is used)
* `[-t | --topic]` to define topic to subscribe to or column in csv file

If you want to stream example dataset use

```bash
python3 dynamic_signal_limits_service.py -f config.ini -t "Average Cell Temperature"
```

where your `config.ini` shall contain

```ini
[file]
path=data/input/average_temperature.csv
```

Now, let's query the latest limits from data/output/dynamic_limits.json

```bash
python3 query_signal_limits.py
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
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## 👐 Contributing

Feel free to contribute in any way you like, we're always open to new ideas and approaches.

* Feel welcome to [open an issue](https://github.com/MarekWadinger/online_outlier_detection/issues/new/choose) if you think you've spotted a bug or a performance issue.

<!-- 
## 🤝 Affiliations

<p align="center">
  <img width="70%" src="" alt="affiliations">
</p>
-->

## 💬 Citation

If the service or the algorithm has been useful to you and you would like to cite it in an scientific publication, please refer to the [paper](https://arxiv.org/abs/2301.13527) currently under review:

```bibtex
@under_review{wadinger_kvasnica_2023, 
  title={Real-time outlier detection with dynamic process limits}, 
  url={https://arxiv.org/abs/2301.13527}, 
  journal={arXiv.org}, 
  author={Wadinger, Marek and Kvasnica, Michal}, 
  year={2023}
} 
```

<!-- 
## 📝 License

This algorithm is free and open-source software licensed under the [3-clause BSD license](https://github.com/online-ml/river/blob/main/LICENSE).
  -->
