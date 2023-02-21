# Real-Time Outlier Detection with Dynamic Process Limits
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

> ![Inverter_Temperature_168_hours_sliding_thresh](https://user-images.githubusercontent.com/50716630/220299639-b3f25288-dac6-428d-9270-6e8125915970.png)
## ⚡️ Quickstart

Quick way to get your hand on the algorithm is to use this [notebook](https://github.com/MarekWadinger/online_outlier_detection/blob/main/online_outlier_detection.ipynb) to play around with example data.

As a quick example, we will run the online service in the terminal and use query the latest dynamic limits for our process
To start the service, run following line of code in your terminal:
```bash
python3 dynamic_signal_limits_service.py
```
Note: You can modify the input data stream using attributes:
* `[-f | --file]` with path to csv file.  This makes the service publish data to the JSON file dynamic_limits.json
* `[-u | --url]` with MQTT host url. This makes the service publish data to the topics parent level + "/dynamic_limits"
* `[-t | --topic]` to define topic to subscribe to or column in csv file

If you want to stream example datasets use
```bash
python3 dynamic_signal_limits_service.py -f "average_temperature.csv" -t "Average Cell Temperature"
```

Now, let's query the latest limits from dynamic_limits.json
```bash
python3 query_signal_limits.py
```
The response is in the format
```python
{'time': datetime.datetime(1970, 1, 1, 14, 52, 42), 
 'anomaly': 0, 
 'level_high': 1180.92, 
 'level_low': 1151.15}
```

## 🛠 Installation
```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

 ## 👐 Contributing

Feel free to contribute in any way you like, we're always open to new ideas and approaches.

- Feel welcome to [open an issue](https://github.com/MarekWadinger/online_outlier_detection/issues/new/choose) if you think you've spotted a bug or a performance issue.

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
